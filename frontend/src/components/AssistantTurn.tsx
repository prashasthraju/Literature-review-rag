import { AlertTriangle, Check, ChevronDown, ChevronRight, Copy, Loader2, Search, ThumbsDown, ThumbsUp } from 'lucide-react'
import { useState } from 'react'
import { answerToMarkdown, pagesLabel, seconds, shortModel, splitMarkers } from '../lib/format'
import type { AssistantMessage, Citation, Source, Turn } from '../types'

interface Props {
  turn: Turn
  activeSource: string | null
  onOpenSource: (source: Source, citations: Citation[]) => void
  onFeedback: (answer: AssistantMessage, rating: 1 | -1) => void
}

const STAGE_TEXT = {
  queued: 'Waiting for the previous question to finish…',
  rewriting: 'Working out what your follow-up refers to…',
  retrieving: 'Searching the papers…',
  generating: 'Reading the passages and writing a cited answer…',
}

const STATUS = {
  answered: null,
  partial: { label: 'Partial answer', tone: 'border-amber-500/30 bg-amber-50 text-amber-800 dark:bg-amber-900/20 dark:text-amber-200' },
  insufficient_evidence: {
    label: 'Not found in the papers',
    tone: 'border-ink-300 bg-ink-100 text-ink-600 dark:border-ink-600 dark:bg-ink-800 dark:text-ink-300',
  },
  invalid_output: { label: 'No valid answer', tone: 'border-red-500/30 bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300' },
}

function uniqueTitles(sources: Source[]) {
  return [...new Set(sources.map((source) => source.document_title))]
}

function Progress({ turn }: { turn: Turn }) {
  const titles = turn.sources ? uniqueTitles(turn.sources) : []
  return (
    <div className="space-y-2 text-sm text-ink-500 dark:text-ink-400">
      {turn.rewritten && (
        <div className="flex items-center gap-1.5 text-xs">
          <Search size={12} /> Searching for “{turn.rewritten}”
        </div>
      )}
      <div className="flex items-center gap-2">
        <Loader2 size={15} className="animate-spin" />
        {turn.stage ? STAGE_TEXT[turn.stage] : 'Sending…'}
      </div>
      {turn.sources && turn.stage === 'generating' && (
        <div className="text-xs">
          {turn.sources.length} passages from {titles.slice(0, 3).join('; ')}
          {titles.length > 3 && ` and ${titles.length - 3} more`}
        </div>
      )}
    </div>
  )
}

function AnswerText({ answer, activeSource, onOpen }: {
  answer: AssistantMessage
  activeSource: string | null
  onOpen: (sourceId: string) => void
}) {
  const verified = new Set(answer.citations.map((citation) => citation.source_id))
  return (
    <div className="font-serif text-[16px] leading-relaxed whitespace-pre-wrap">
      {splitMarkers(answer.answer).map((segment, index) => {
        if (segment.kind === 'text') return <span key={index}>{segment.value}</span>
        const ok = verified.has(segment.sourceId)
        return (
          <button
            key={index}
            onClick={() => onOpen(segment.sourceId)}
            title={ok ? 'Open the cited passage' : 'No verified quote for this source'}
            className={`mx-0.5 inline-flex -translate-y-px items-center rounded px-1 font-sans text-[11px] font-semibold align-middle ${
              ok
                ? activeSource === segment.sourceId
                  ? 'bg-accent-600 text-white'
                  : 'bg-accent-100 text-accent-700 hover:bg-accent-500 hover:text-white dark:bg-accent-700/60 dark:text-accent-100'
                : 'bg-ink-200 text-ink-500 line-through dark:bg-ink-700'
            }`}
          >
            {segment.sourceId}
          </button>
        )
      })}
    </div>
  )
}

export function AssistantTurn({ turn, activeSource, onOpenSource, onFeedback }: Props) {
  const [showAll, setShowAll] = useState(false)
  const [copied, setCopied] = useState(false)
  const answer = turn.answer

  if (turn.error) {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-300">
        <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {turn.error}
      </div>
    )
  }
  if (!answer) return <Progress turn={turn} />

  const sourceById = new Map(answer.sources.map((source) => [source.source_id, source]))
  // A source can be quoted more than once; show one card per source, in citation order.
  const citationsBySource = new Map<string, Citation[]>()
  for (const citation of answer.citations) {
    citationsBySource.set(citation.source_id, [...(citationsBySource.get(citation.source_id) ?? []), citation])
  }
  const open = (sourceId: string) => {
    const source = sourceById.get(sourceId)
    if (source) onOpenSource(source, citationsBySource.get(sourceId) ?? [])
  }
  const uncited = answer.sources.filter((source) => !citationsBySource.has(source.source_id))
  const status = STATUS[answer.status]
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(answerToMarkdown(answer))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard unavailable (e.g. insecure context); nothing to do
    }
  }

  return (
    <div className="space-y-4">
      {answer.standalone_question !== answer.question && (
        <div className="flex items-center gap-1.5 text-xs text-ink-500 dark:text-ink-400">
          <Search size={12} /> Searched for “{answer.standalone_question}”
        </div>
      )}
      {status && (
        <span className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-medium ${status.tone}`}>{status.label}</span>
      )}

      <AnswerText answer={answer} activeSource={activeSource} onOpen={open} />

      {answer.caveats.length > 0 && (
        <ul className="space-y-1 rounded-lg border border-amber-500/30 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-900/20 dark:text-amber-100">
          {answer.caveats.map((caveat, index) => (
            <li key={index} className="flex gap-2">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {caveat}
            </li>
          ))}
        </ul>
      )}

      {answer.citations.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-semibold tracking-wide text-ink-500 uppercase dark:text-ink-400">
            Verified sources
          </div>
          {[...citationsBySource.entries()].map(([sourceId, quotes]) => {
            const citation = quotes[0]
            return (
            <button
              key={sourceId}
              onClick={() => open(sourceId)}
              className={`block w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                activeSource === sourceId
                  ? 'border-accent-500 bg-accent-50 dark:bg-accent-700/30'
                  : 'border-ink-200 bg-white hover:border-accent-500/50 dark:border-ink-700 dark:bg-ink-800'
              }`}
            >
              <div className="flex items-baseline gap-2 text-sm">
                <span className="font-semibold text-accent-600 dark:text-accent-100">{citation.source_id}</span>
                <span className="min-w-0 flex-1 truncate font-medium">{citation.document_title}</span>
                <span className="shrink-0 text-xs text-ink-500">{pagesLabel(citation.pages)}</span>
              </div>
              <div className="mt-0.5 truncate text-xs text-ink-500 dark:text-ink-400">{citation.section_title}</div>
              {quotes.map((quote, index) => (
                <div key={index} className="mt-1 font-serif text-sm text-ink-700 italic dark:text-ink-200">
                  “{quote.quote}”
                  {quote.match === 'approximate' && (
                    <span className="ml-1 font-sans text-[11px] text-ink-500 not-italic">(near-verbatim; numbers match)</span>
                  )}
                </div>
              ))}
            </button>
            )
          })}
        </div>
      )}

      {answer.unverified_citations > 0 && (
        <p className="text-xs text-ink-500 dark:text-ink-400">
          {answer.unverified_citations} citation{answer.unverified_citations > 1 ? 's' : ''} could not be verified against
          the cited passage and {answer.unverified_citations > 1 ? 'were' : 'was'} removed.
        </p>
      )}

      {uncited.length > 0 && (
        <div>
          <button
            onClick={() => setShowAll(!showAll)}
            className="flex items-center gap-1 text-xs text-ink-500 hover:text-ink-700 dark:text-ink-400 dark:hover:text-ink-200"
          >
            {showAll ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {answer.citations.length ? 'Other passages read' : 'Passages searched'} ({uncited.length})
          </button>
          {showAll && (
            <ul className="mt-2 space-y-1">
              {uncited.map((source) => (
                <li key={source.source_id}>
                  <button
                    onClick={() => onOpenSource(source, [])}
                    className="flex w-full gap-2 rounded px-2 py-1 text-left text-xs hover:bg-ink-100 dark:hover:bg-ink-800"
                  >
                    <span className="font-semibold text-ink-500">{source.source_id}</span>
                    <span className="min-w-0 flex-1 truncate">
                      {source.document_title} — {source.section_title}
                    </span>
                    <span className="shrink-0 text-ink-500">{pagesLabel(source.pages)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="flex items-center gap-1 text-xs text-ink-400">
        <span title={Object.entries(answer.timings_ms).map(([stage, ms]) => `${stage}: ${seconds(ms)}`).join('\n')}>
          {seconds(answer.timings_ms.total)} · {shortModel(answer.model)}
        </span>
        <span className="flex-1" />
        <button onClick={copy} className="rounded p-1.5 hover:bg-ink-100 hover:text-ink-700 dark:hover:bg-ink-800" title="Copy with sources">
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
        <button
          onClick={() => onFeedback(answer, 1)}
          className={`rounded p-1.5 hover:bg-ink-100 dark:hover:bg-ink-800 ${answer.feedback === 1 ? 'text-emerald-600' : ''}`}
          title="Good answer"
        >
          <ThumbsUp size={14} />
        </button>
        <button
          onClick={() => onFeedback(answer, -1)}
          className={`rounded p-1.5 hover:bg-ink-100 dark:hover:bg-ink-800 ${answer.feedback === -1 ? 'text-red-600' : ''}`}
          title="Wrong or unhelpful answer"
        >
          <ThumbsDown size={14} />
        </button>
      </div>
    </div>
  )
}
