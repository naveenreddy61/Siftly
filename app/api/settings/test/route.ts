import { NextResponse } from 'next/server'
import { getGeminiClient } from '@/lib/gemini-client'

export async function POST(): Promise<NextResponse> {
  const client = await getGeminiClient()
  if (!client) {
    return NextResponse.json({ working: false, error: 'No Gemini API key configured' })
  }

  try {
    const result = await client.generateContent('Say hi in one word')
    if (result.response && result.response.text()) {
      return NextResponse.json({ working: true })
    }
    return NextResponse.json({ working: false, error: 'Empty response from Gemini' })
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    const friendly = msg.includes('401') || msg.includes('API_KEY_INVALID')
      ? 'Invalid API key'
      : msg.includes('403')
      ? 'API key does not have permission'
      : msg.slice(0, 120)
    return NextResponse.json({ working: false, error: friendly })
  }
}
