import type {
  AssistantMessage,
  ConversationDetail,
  ConversationSummary,
  Health,
  Paper,
  Source,
  Stage,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) throw new Error(await errorMessage(response))
  return (response.status === 204 ? undefined : await response.json()) as T
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((item) => item.msg).join('; ')
    if (detail?.message) return detail.message
  } catch {
    // fall through to the status text
  }
  return `${response.status} ${response.statusText}`
}

export const api = {
  health: () => request<Health>('/api/health'),
  papers: () => request<Paper[]>('/api/papers'),
  conversations: () => request<ConversationSummary[]>('/api/conversations'),
  conversation: (id: string) => request<ConversationDetail>(`/api/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/api/conversations/${id}`, { method: 'DELETE' }),
  feedback: (messageId: string, rating: 1 | -1) =>
    request<void>(`/api/messages/${messageId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating }),
    }),
}

export const pdfUrl = (paperId: string, page?: number) =>
  `/api/papers/${encodeURIComponent(paperId)}/pdf${page ? `#page=${page}` : ''}`

export type ChatEvent =
  | { event: 'meta'; data: { request_id: string; conversation_id: string; user_message_id: string; message_id: string } }
  | { event: 'stage'; data: { stage: Stage } }
  | { event: 'rewritten'; data: { standalone_question: string } }
  | { event: 'sources'; data: { sources: Source[] } }
  | { event: 'answer'; data: AssistantMessage }
  | { event: 'error'; data: { message: string; request_id: string } }

export interface ChatInput {
  message: string
  conversation_id: string | null
  paper_ids: string[]
}

/** POST a chat turn and call onEvent for each server-sent event. */
export async function streamChat(input: ChatInput, onEvent: (event: ChatEvent) => void, signal: AbortSignal) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  })
  if (!response.ok || !response.body) throw new Error(await errorMessage(response))
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7)
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      if (data) onEvent({ event, data: JSON.parse(data) } as ChatEvent)
    }
  }
}
