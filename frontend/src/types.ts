// Mirrors src/api/schemas.py.

export type Status = 'answered' | 'partial' | 'insufficient_evidence' | 'invalid_output'

export interface Citation {
  source_id: string
  quote: string
  match: 'exact' | 'approximate' | 'none'
  document_id: string | null
  document_title: string | null
  section_title: string | null
  pages: number[]
}

export interface Source {
  source_id: string
  document_id: string
  document_title: string
  section_title: string
  pages: number[]
  text: string
  cited: boolean
}

export interface AssistantMessage {
  message_id: string
  conversation_id: string
  request_id: string
  question: string
  standalone_question: string
  paper_ids: string[]
  status: Status
  answer: string
  citations: Citation[]
  sources: Source[]
  caveats: string[]
  unverified_citations: number
  citation_coverage: number | null
  model: string
  prompt_version: string
  timings_ms: Record<string, number>
  created_at: string
  feedback: number | null
}

export interface MessageView {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  answer: AssistantMessage | null
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ConversationDetail extends ConversationSummary {
  messages: MessageView[]
}

export interface Paper {
  id: string
  title: string
  topic: string
  chunks: number
  pages: number
  sections: string[]
  has_pdf: boolean
}

export interface Health {
  status: 'loading' | 'ready' | 'error'
  detail: string | null
  generator: string
  retrieval_config: string
  prompt_version: string
  papers: number
}

export type Stage = 'queued' | 'rewriting' | 'retrieving' | 'generating'

export interface Turn {
  id: string
  question: string
  stage?: Stage
  rewritten?: string
  sources?: Source[]
  answer?: AssistantMessage
  error?: string
}
