import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/db'

function maskKey(raw: string | null): string | null {
  if (!raw) return null
  if (raw.length <= 8) return '********'
  return `${raw.slice(0, 6)}${'*'.repeat(raw.length - 10)}${raw.slice(-4)}`
}

const ALLOWED_GEMINI_MODELS = [
  'gemini-3.1-flash-lite-preview',
] as const

export async function GET(): Promise<NextResponse> {
  try {
    const [gemini, geminiModel] = await Promise.all([
      prisma.setting.findUnique({ where: { key: 'geminiApiKey' } }),
      prisma.setting.findUnique({ where: { key: 'geminiModel' } }),
    ])

    return NextResponse.json({
      geminiApiKey: maskKey(gemini?.value ?? null),
      hasGeminiKey: gemini !== null,
      geminiModel: geminiModel?.value ?? 'gemini-3.1-flash-lite-preview',
    })
  } catch (err) {
    console.error('Settings GET error:', err)
    return NextResponse.json(
      { error: `Failed to fetch settings: ${err instanceof Error ? err.message : String(err)}` },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: {
    geminiApiKey?: string
    geminiModel?: string
  } = {}
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { geminiApiKey, geminiModel } = body

  // Save Gemini model if provided
  if (geminiModel !== undefined) {
    if (!(ALLOWED_GEMINI_MODELS as readonly string[]).includes(geminiModel)) {
      return NextResponse.json({ error: 'Invalid Gemini model' }, { status: 400 })
    }
    await prisma.setting.upsert({
      where: { key: 'geminiModel' },
      update: { value: geminiModel },
      create: { key: 'geminiModel', value: geminiModel },
    })
    return NextResponse.json({ saved: true })
  }

  // Save Gemini key if provided
  if (geminiApiKey !== undefined) {
    if (typeof geminiApiKey !== 'string' || geminiApiKey.trim() === '') {
      return NextResponse.json({ error: 'Invalid geminiApiKey value' }, { status: 400 })
    }
    const trimmed = geminiApiKey.trim()
    try {
      await prisma.setting.upsert({
        where: { key: 'geminiApiKey' },
        update: { value: trimmed },
        create: { key: 'geminiApiKey', value: trimmed },
      })
      return NextResponse.json({ saved: true })
    } catch (err) {
      console.error('Settings POST error:', err)
      return NextResponse.json(
        { error: `Failed to save: ${err instanceof Error ? err.message : String(err)}` },
        { status: 500 }
      )
    }
  }

  return NextResponse.json({ error: 'No setting provided' }, { status: 400 })
}

export async function DELETE(request: NextRequest): Promise<NextResponse> {
  let body: { key?: string } = {}
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const allowed = ['geminiApiKey', 'geminiModel']
  if (!body.key || !allowed.includes(body.key)) {
    return NextResponse.json({ error: 'Invalid key' }, { status: 400 })
  }

  await prisma.setting.deleteMany({ where: { key: body.key } })
  return NextResponse.json({ deleted: true })
}
