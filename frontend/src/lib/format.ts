import type { AssistantMessage, Turn } from '../types'

export type Segment = { kind: 'text'; value: string } | { kind: 'marker'; sourceId: string }

const MARKER_RE = /\[(S\d+)\]/g

/** Split answer text into prose and [S#] citation markers. */
export function splitMarkers(text: string): Segment[] {
  const segments: Segment[] = []
  let last = 0
  for (const match of text.matchAll(MARKER_RE)) {
    if (match.index > last) segments.push({ kind: 'text', value: text.slice(last, match.index) })
    segments.push({ kind: 'marker', sourceId: match[1] })
    last = match.index + match[0].length
  }
  if (last < text.length) segments.push({ kind: 'text', value: text.slice(last) })
  return segments
}

const escape = (word: string) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Character ranges of a quote inside a passage, tolerant of spacing and punctuation. */
export function quoteRanges(text: string, quote: string): [number, number][] {
  const ranges: [number, number][] = []
  for (const fragment of quote.split(/\s*(?:\.\.\.|…|\n)\s*/)) {
    const words = fragment.match(/[\p{L}\p{N}]+/gu)
    if (!words || words.length < 2) continue
    const pattern = new RegExp(words.map(escape).join('[^\\p{L}\\p{N}]+'), 'iu')
    const match = pattern.exec(text)
    if (match) ranges.push([match.index, match.index + match[0].length])
  }
  return ranges.sort((a, b) => a[0] - b[0])
}

export function pagesLabel(pages: number[]): string {
  if (!pages.length) return ''
  return pages.length === 1 ? `p. ${pages[0]}` : `pp. ${pages[0]}–${pages[pages.length - 1]}`
}

export function shortModel(name: string): string {
  return name.replace(/^[a-z]+:/, '').replace(/@.*$/, '').split('/').pop() ?? name
}

export function seconds(ms: number | undefined): string {
  return ms === undefined ? '' : `${(ms / 1000).toFixed(1)} s`
}

export function answerToMarkdown(answer: AssistantMessage): string {
  const lines = [answer.answer]
  if (answer.caveats.length) lines.push('', ...answer.caveats.map((caveat) => `> Note: ${caveat}`))
  if (answer.citations.length) {
    lines.push('', '**Sources**')
    for (const citation of answer.citations) {
      const where = [citation.section_title, pagesLabel(citation.pages)].filter(Boolean).join(', ')
      lines.push(`- [${citation.source_id}] *${citation.document_title}* — ${where}: “${citation.quote}”`)
    }
  }
  return lines.join('\n')
}

export function conversationToMarkdown(title: string, turns: Turn[]): string {
  const parts = [`# ${title}`, '']
  for (const turn of turns) {
    parts.push(`## ${turn.question}`, '')
    if (turn.answer) parts.push(answerToMarkdown(turn.answer), '')
    else if (turn.error) parts.push(`*${turn.error}*`, '')
  }
  return parts.join('\n')
}

export function newId(): string {
  return typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : Math.random().toString(36).slice(2)
}
