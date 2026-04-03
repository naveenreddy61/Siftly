import { GenerativeModel } from '@google/generative-ai'
import prisma from '@/lib/db'
import { buildImageContext } from '@/lib/image-context'
import { getGeminiClient, getGeminiModel } from '@/lib/gemini-client'

type AllowedMediaType = 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp'

function guessMediaType(url: string, contentTypeHeader: string | null): AllowedMediaType {
  const ct = contentTypeHeader?.toLowerCase() ?? ''
  if (ct.includes('png') || url.includes('.png')) return 'image/png'
  if (ct.includes('gif') || url.includes('.gif')) return 'image/gif'
  if (ct.includes('webp') || url.includes('.webp')) return 'image/webp'
  return 'image/jpeg'
}

const MAX_IMAGE_BYTES = 3_500_000 // 3.5MB raw

async function fetchImageAsArrayBuffer(url: string): Promise<ArrayBuffer | null> {
  try {
    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        Referer: 'https://twitter.com/',
      },
      signal: AbortSignal.timeout(4000),
    })
    if (!res.ok) return null
    const buffer = await res.arrayBuffer()
    if (buffer.byteLength < 500) return null
    if (buffer.byteLength > MAX_IMAGE_BYTES) {
      console.warn(`[vision] skipping oversized image (${Math.round(buffer.byteLength / 1024)}KB): ${url.slice(0, 80)}`)
      return null
    }
    return buffer
  } catch {
    return null
  }
}

const ANALYSIS_PROMPT = `Analyze this image for a bookmark search system. Return ONLY valid JSON, no markdown, no explanation.

{
  "people": ["description of each person visible — age, gender, appearance, expression, what they're doing"],
  "text_ocr": ["ALL visible text exactly as written — signs, captions, UI text, meme text, headlines, code"],
  "objects": ["significant objects, brands, logos, symbols, technology"],
  "scene": "brief scene description — setting and platform (e.g. 'Twitter screenshot', 'office desk', 'terminal window')",
  "action": "what is happening or being shown",
  "mood": "emotional tone: humorous/educational/alarming/inspiring/satirical/celebratory/neutral",
  "style": "photo/screenshot/meme/chart/infographic/artwork/gif/code/diagram",
  "meme_template": "specific meme template name if applicable, else null",
  "tags": ["30-40 specific searchable tags — topics, synonyms, proper nouns, brands, actions, emotions"]
}

Rules:
- text_ocr: transcribe ALL readable text exactly, word for word
- If a financial chart: include asset name, direction (up/down), timeframe
- If code: include language, key function/concept names
- If a meme: include the exact template name
- tags: be maximally specific — include brand names, person names, tool names, technical terms
- BAD tags: "twitter", "post", "image", "screenshot" (too generic)
- GOOD tags: "bitcoin price chart", "react hooks", "frustrated man", "gpt-4", "bull market"`

const RETRY_DELAYS_MS = [1500, 4000, 10000]
const CONCURRENCY = 12

async function analyzeImageWithRetry(
  url: string,
  model: GenerativeModel,
  attempt = 0,
): Promise<string> {
  const img = await fetchImageAsArrayBuffer(url)
  if (!img) return ''

  try {
    const result = await model.generateContent([
      {
        inlineData: {
          mimeType: guessMediaType(url, null),
          data: Buffer.from(img).toString('base64'),
        },
      },
      ANALYSIS_PROMPT,
    ])
    const raw = result.response.text()?.trim() ?? ''
    if (!raw) return ''

    const jsonMatch = raw.match(/\{[\s\S]*\}/)
    if (!jsonMatch) return ''
    JSON.parse(jsonMatch[0])
    return jsonMatch[0]
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    const isClientError = msg.includes('400') || msg.includes('401') || msg.includes('403') || msg.includes('422')
    const isRetryable = !isClientError

    if (attempt === 0) {
      console.warn(`[vision] Gemini analysis failed (attempt ${attempt + 1}): ${msg.slice(0, 400)}`)
    }

    if (isRetryable && attempt < RETRY_DELAYS_MS.length) {
      await new Promise((r) => setTimeout(r, RETRY_DELAYS_MS[attempt]))
      return analyzeImageWithRetry(url, model, attempt + 1)
    }
    return ''
  }
}

export interface MediaItemForAnalysis {
  id: string
  url: string
  thumbnailUrl: string | null
  type: string
}

async function getCachedAnalysis(imageUrl: string, excludeId: string): Promise<string | null> {
  const existing = await prisma.mediaItem.findFirst({
    where: { url: imageUrl, imageTags: { not: null }, id: { not: excludeId } },
    select: { imageTags: true },
  })
  return existing?.imageTags ?? null
}

export async function analyzeItem(
  item: MediaItemForAnalysis,
  geminiModel: GenerativeModel,
): Promise<number> {
  const imageUrl = item.type === 'video' ? (item.thumbnailUrl ?? item.url) : item.url

  const cached = await getCachedAnalysis(imageUrl, item.id)
  if (cached) {
    await prisma.mediaItem.update({ where: { id: item.id }, data: { imageTags: cached } })
    return 1
  }

  const prefix = item.type === 'video' ? '{"_type":"video_thumbnail",' : ''
  let tags = await analyzeImageWithRetry(imageUrl, geminiModel)

  if (tags && prefix) {
    tags = tags.replace(/^\{/, prefix)
  }

  if (tags) {
    await prisma.mediaItem.update({ where: { id: item.id }, data: { imageTags: tags } })
    return 1
  }

  await prisma.mediaItem.update({ where: { id: item.id }, data: { imageTags: '{}' } })
  return 0
}

export async function runWithConcurrency<T>(
  tasks: (() => Promise<T>)[],
  limit: number,
): Promise<T[]> {
  const results: T[] = []
  let index = 0

  async function worker() {
    while (index < tasks.length) {
      const taskIndex = index++
      results[taskIndex] = await tasks[taskIndex]()
    }
  }

  const workers = Array.from({ length: Math.min(limit, tasks.length) }, () => worker())
  await Promise.all(workers)
  return results
}

export async function analyzeBatch(
  items: MediaItemForAnalysis[],
  geminiModel: GenerativeModel,
  onProgress?: (delta: number) => void,
  shouldAbort?: () => boolean,
): Promise<number> {
  const analyzable = items.filter((m) => m.type === 'photo' || m.type === 'gif' || m.type === 'video')
  if (analyzable.length === 0) return 0

  const tasks = analyzable.map((item) => async () => {
    if (shouldAbort?.()) return 0
    const result = await analyzeItem(item, geminiModel)
    onProgress?.(1)
    return result
  })
  const results = await runWithConcurrency(tasks, CONCURRENCY)

  return results.reduce((sum, r) => sum + r, 0)
}

export async function analyzeUntaggedImages(
  geminiModel: GenerativeModel,
  limit = 10,
): Promise<number> {
  const untagged = await prisma.mediaItem.findMany({
    where: { imageTags: null, type: { in: ['photo', 'gif', 'video'] } },
    take: limit,
    select: { id: true, url: true, thumbnailUrl: true, type: true },
  })
  if (untagged.length === 0) return 0
  return analyzeBatch(untagged, geminiModel)
}

export async function analyzeAllUntagged(
  geminiModel: GenerativeModel,
  onProgress?: (total: number) => void,
  shouldAbort?: () => boolean,
): Promise<number> {
  const CHUNK = 15
  let total = 0
  let cursor: string | undefined

  while (true) {
    if (shouldAbort?.()) break

    const untagged = await prisma.mediaItem.findMany({
      where: {
        type: { in: ['photo', 'gif', 'video'] },
        imageTags: null,
        ...(cursor ? { id: { gt: cursor } } : {}),
      },
      orderBy: { id: 'asc' },
      take: CHUNK,
      select: { id: true, url: true, thumbnailUrl: true, type: true },
    })

    if (untagged.length === 0) break

    cursor = untagged[untagged.length - 1].id

    await analyzeBatch(untagged, geminiModel, (delta) => {
      total += delta
      onProgress?.(total)
    }, shouldAbort)

    if (untagged.length < CHUNK) break
  }

  return total
}

// ── Batch semantic enrichment ──────────────────────────────────────────────────

const ENRICH_BATCH_SIZE = 5
const ENRICH_CONCURRENCY = 2

export interface BookmarkForEnrichment {
  id: string
  text: string
  imageTags: string[]
  entities?: {
    hashtags?: string[]
    urls?: string[]
    mentions?: string[]
    tools?: string[]
    tweetType?: string
  }
}

export interface EnrichmentResult {
  id: string
  tags: string[]
  sentiment: string
  people: string[]
  companies: string[]
}

function buildEnrichmentPrompt(bookmarks: BookmarkForEnrichment[]): string {
  const items = bookmarks.map((b) => {
    const entry: Record<string, unknown> = { id: b.id, text: b.text.slice(0, 500) }
    const imgCtx = b.imageTags.map((raw) => buildImageContext(raw)).filter(Boolean).join(' | ')
    if (imgCtx) entry.imageContext = imgCtx
    if (b.entities?.hashtags?.length) entry.hashtags = b.entities.hashtags.slice(0, 8)
    if (b.entities?.tools?.length) entry.tools = b.entities.tools
    if (b.entities?.mentions?.length) entry.mentions = b.entities.mentions.slice(0, 3)
    return entry
  })

  return `Generate search tags and metadata for each of these Twitter/X bookmarks.

For each bookmark return:
- tags: 25-35 specific semantic search tags covering entities, actions, visual content, synonyms, and emotional signals
- sentiment: one of "positive", "negative", "neutral", "humorous", "controversial"
- people: named people mentioned or shown (max 5, empty array if none)
- companies: company/product/tool names explicitly referenced (max 8, empty array if none)

Rules for tags:
- 2-5 words max, specific beats generic
- NO generic terms: "twitter post", "screenshot", "social media", "content"
- YES to proper nouns, version numbers, specific concepts
- Rank most-search-relevant tags first

Return ONLY valid JSON, no markdown:
[{"id":"...","tags":[...],"sentiment":"...","people":[...],"companies":[...]}]

BOOKMARKS:
${JSON.stringify(items, null, 1)}`
}

export async function enrichBatchSemanticTags(
  bookmarks: BookmarkForEnrichment[],
  geminiModel: GenerativeModel,
): Promise<EnrichmentResult[]> {
  if (bookmarks.length === 0) return []

  const prompt = buildEnrichmentPrompt(bookmarks)

  const parseResponse = (text: string): EnrichmentResult[] => {
    const match = text.match(/\[[\s\S]*\]/)
    if (!match) return []
    const parsed: unknown = JSON.parse(match[0])
    if (!Array.isArray(parsed)) return []
    return (parsed as Record<string, unknown>[]).map((item): EnrichmentResult => ({
      id: String(item.id ?? ''),
      tags: Array.isArray(item.tags) ? (item.tags as unknown[]).map(String).filter(Boolean) : [],
      sentiment: String(item.sentiment ?? 'neutral'),
      people: Array.isArray(item.people) ? (item.people as unknown[]).map(String).filter(Boolean) : [],
      companies: Array.isArray(item.companies) ? (item.companies as unknown[]).map(String).filter(Boolean) : [],
    })).filter((r) => r.id)
  }

  const ENRICH_RETRY_DELAYS = [2000, 5000]
  for (let attempt = 0; attempt <= ENRICH_RETRY_DELAYS.length; attempt++) {
    try {
      const result = await geminiModel.generateContent(prompt)
      const text = result.response.text() ?? ''
      const results = parseResponse(text)
      if (results.length > 0) return results
      console.warn(`[enrich] Gemini no JSON array in response (attempt ${attempt + 1})`)
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err)
      console.warn(`[enrich] Gemini batch failed (attempt ${attempt + 1}): ${errMsg.slice(0, 400)}`)
      const isClientError = errMsg.includes('400') || errMsg.includes('401') || errMsg.includes('403') || errMsg.includes('422')
      if (isClientError || attempt >= ENRICH_RETRY_DELAYS.length) break
      await new Promise((r) => setTimeout(r, ENRICH_RETRY_DELAYS[attempt]))
    }
  }
  return []
}

export async function enrichAllBookmarks(
  geminiModel: GenerativeModel,
  onProgress?: (total: number) => void,
  shouldAbort?: () => boolean,
): Promise<number> {
  const CHUNK = ENRICH_BATCH_SIZE * ENRICH_CONCURRENCY * 2
  let enriched = 0
  let cursor: string | undefined

  while (true) {
    if (shouldAbort?.()) break

    const rows = await prisma.bookmark.findMany({
      where: {
        semanticTags: null,
        ...(cursor ? { id: { gt: cursor } } : {}),
      },
      orderBy: { id: 'asc' },
      take: CHUNK,
      select: {
        id: true,
        text: true,
        entities: true,
        mediaItems: { select: { imageTags: true } },
      },
    })

    if (rows.length === 0) break
    cursor = rows[rows.length - 1].id

    const trivialIds: string[] = []
    const toEnrich: BookmarkForEnrichment[] = []

    for (const b of rows) {
      const imageTags = b.mediaItems
        .map((m) => m.imageTags)
        .filter((t): t is string => t !== null && t !== '' && t !== '{}')

      if (imageTags.length === 0 && b.text.length < 20) {
        trivialIds.push(b.id)
        continue
      }

      let entities: BookmarkForEnrichment['entities'] = undefined
      if (b.entities) {
        try { entities = JSON.parse(b.entities) as typeof entities } catch { /* ignore */ }
      }

      toEnrich.push({ id: b.id, text: b.text, imageTags, entities })
    }

    if (trivialIds.length > 0) {
      await prisma.bookmark.updateMany({
        where: { id: { in: trivialIds } },
        data: { semanticTags: '[]' },
      })
    }

    const batches: BookmarkForEnrichment[][] = []
    for (let i = 0; i < toEnrich.length; i += ENRICH_BATCH_SIZE) {
      batches.push(toEnrich.slice(i, i + ENRICH_BATCH_SIZE))
    }

    const batchTasks = batches.map((batch) => async () => {
      if (shouldAbort?.()) return

      const results = await enrichBatchSemanticTags(batch, geminiModel)
      const resultMap = new Map(results.map((r) => [r.id, r]))

      for (const b of batch) {
        const result = resultMap.get(b.id)
        if (result?.tags.length) {
          await prisma.bookmark.update({
            where: { id: b.id },
            data: {
              semanticTags: JSON.stringify(result.tags),
              enrichmentMeta: JSON.stringify({
                sentiment: result.sentiment,
                people: result.people,
                companies: result.companies,
              }),
            },
          })
          enriched++
          onProgress?.(enriched)
        }
      }
    })

    await runWithConcurrency(batchTasks, ENRICH_CONCURRENCY)

    if (rows.length < CHUNK) break
  }

  return enriched
}

export async function enrichBookmarkSemanticTags(
  bookmarkId: string,
  tweetText: string,
  imageTags: string[],
  geminiModel: GenerativeModel,
  entities?: BookmarkForEnrichment['entities'],
): Promise<string[]> {
  const results = await enrichBatchSemanticTags(
    [{ id: bookmarkId, text: tweetText, imageTags, entities }],
    geminiModel,
  )
  const result = results[0]
  if (!result?.tags.length) return []

  await prisma.bookmark.update({
    where: { id: bookmarkId },
    data: {
      semanticTags: JSON.stringify(result.tags),
      enrichmentMeta: JSON.stringify({
        sentiment: result.sentiment,
        people: result.people,
        companies: result.companies,
      }),
    },
  })
  return result.tags
}

// suppress unused import warnings
void getGeminiModel
