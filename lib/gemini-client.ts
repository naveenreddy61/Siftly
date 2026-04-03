import { GoogleGenerativeAI, GenerativeModel, SchemaType } from '@google/generative-ai'
import prisma from '@/lib/db'

// Module-level model cache — avoids hundreds of DB roundtrips per pipeline run
let _cachedModel: string | null = null
let _modelCacheExpiry = 0

/**
 * Get the configured Gemini model from settings (cached for 5 minutes).
 */
export async function getGeminiModel(): Promise<string> {
  if (_cachedModel && Date.now() < _modelCacheExpiry) return _cachedModel
  const setting = await prisma.setting.findUnique({ where: { key: 'geminiModel' } })
  _cachedModel = setting?.value ?? 'gemini-3.1-flash-lite-preview'
  _modelCacheExpiry = Date.now() + 5 * 60 * 1000
  return _cachedModel!
}

/**
 * Get the Gemini API key from settings or environment.
 */
export async function getGeminiApiKey(): Promise<string | null> {
  // Check database first
  const setting = await prisma.setting.findUnique({ where: { key: 'geminiApiKey' } })
  if (setting?.value) return setting.value
  
  // Fall back to environment variable
  return process.env.GOOGLE_GENERATIVE_AI_API_KEY || process.env.GEMINI_API_KEY || null
}

/**
 * Initialize and return a Gemini client.
 * Returns null if no API key is configured.
 */
export async function getGeminiClient(): Promise<GenerativeModel | null> {
  const apiKey = await getGeminiApiKey()
  if (!apiKey) return null
  
  const genAI = new GoogleGenerativeAI(apiKey)
  const modelName = await getGeminiModel()
  return genAI.getGenerativeModel({ model: modelName })
}

/**
 * Check if Gemini is configured (API key available).
 */
export async function isGeminiConfigured(): Promise<boolean> {
  const apiKey = await getGeminiApiKey()
  return apiKey !== null
}

/**
 * Get the active AI provider.
 * Always returns 'gemini' since it's the only supported provider.
 */
export async function getActiveProvider(): Promise<'gemini'> {
  return 'gemini'
}
