import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/db'
import { analyzeBatch } from '@/lib/vision-analyzer'
import { getGeminiClient } from '@/lib/gemini-client'

// GET: returns progress stats
export async function GET(): Promise<NextResponse> {
  const [total, tagged] = await Promise.all([
    prisma.mediaItem.count({ where: { type: { in: ['photo', 'gif'] } } }),
    prisma.mediaItem.count({ where: { type: { in: ['photo', 'gif'] }, imageTags: { not: null } } }),
  ])
  return NextResponse.json({ total, tagged, remaining: total - tagged })
}

// POST: analyze a batch of untagged images
export async function POST(request: NextRequest): Promise<NextResponse> {
  let batchSize = 20
  try {
    const body = await request.json()
    if (typeof body.batchSize === 'number') batchSize = Math.min(body.batchSize, 50)
  } catch {
    // use default
  }

  // Get Gemini client (only supported provider)
  const geminiModel = await getGeminiClient()
  
  if (!geminiModel) {
    return NextResponse.json({ error: 'No Gemini API key configured. Add your key in Settings.' }, { status: 400 })
  }

  return runAnalysis(geminiModel, batchSize)
}

async function runAnalysis(
  geminiModel: ReturnType<typeof getGeminiClient> extends Promise<infer T> ? T : never,
  batchSize: number,
): Promise<NextResponse> {
  const untagged = await prisma.mediaItem.findMany({
    where: { imageTags: null, type: { in: ['photo', 'gif'] } },
    take: batchSize,
    select: { id: true, url: true, thumbnailUrl: true, type: true },
  })

  if (untagged.length === 0) {
    return NextResponse.json({ analyzed: 0, remaining: 0, message: 'All images already analyzed.' })
  }

  const analyzed = await analyzeBatch(untagged, null, geminiModel)

  const remaining = await prisma.mediaItem.count({
    where: { imageTags: null, type: { in: ['photo', 'gif'] } },
  })

  return NextResponse.json({ analyzed, remaining })
}
