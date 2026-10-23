import { ExternalLink } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { pdfUrl } from '../api'
import { pagesLabel, quoteRanges } from '../lib/format'
import type { Citation, Source } from '../types'

interface Props {
  source: Source
  citations: Citation[]
}

function Highlighted({ text, quotes }: { text: string; quotes: string[] }) {
  const ranges = quotes.flatMap((quote) => quoteRanges(text, quote)).sort((a, b) => a[0] - b[0])
  if (!ranges.length) return <>{text}</>
  const parts = []
  let last = 0
  ranges.forEach(([start, end], index) => {
    if (start < last) return
    parts.push(text.slice(last, start))
    parts.push(<mark key={index}>{text.slice(start, end)}</mark>)
    last = end
  })
  parts.push(text.slice(last))
  return <>{parts}</>
}

export function SourcePanel({ source, citations }: Props) {
  const [tab, setTab] = useState<'passage' | 'pdf'>('passage')
  const passage = useRef<HTMLDivElement>(null)
  const page = source.pages[0]

  // Bring the first highlighted quote into view.
  useEffect(() => {
    if (tab === 'passage') passage.current?.querySelector('mark')?.scrollIntoView({ block: 'center' })
  }, [tab, source])

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-ink-200 px-4 py-3 dark:border-ink-700">
        <div className="font-serif text-[15px] leading-snug font-semibold">{source.document_title}</div>
        <div className="mt-0.5 text-xs text-ink-500 dark:text-ink-400">
          {source.section_title}
          {source.pages.length > 0 && ` · ${pagesLabel(source.pages)}`}
        </div>
        <div className="mt-3 flex items-center gap-1 text-sm">
          {(['passage', 'pdf'] as const).map((name) => (
            <button
              key={name}
              onClick={() => setTab(name)}
              className={`rounded-md px-2.5 py-1 ${
                tab === name ? 'bg-ink-200 font-medium dark:bg-ink-700' : 'text-ink-500 hover:bg-ink-100 dark:hover:bg-ink-800'
              }`}
            >
              {name === 'passage' ? 'Passage' : 'PDF'}
            </button>
          ))}
          <a
            href={pdfUrl(source.document_id, page)}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-xs text-accent-600 hover:underline dark:text-accent-100"
          >
            Open PDF{page ? ` at p. ${page}` : ''} <ExternalLink size={12} />
          </a>
        </div>
      </div>

      {tab === 'passage' ? (
        <div className="scrollbar-thin flex-1 overflow-y-auto px-4 py-4">
          {citations.length > 0 && (
            <div className="mb-4 space-y-1.5 rounded-lg border border-emerald-600/20 bg-emerald-50 px-3 py-2 text-sm dark:bg-emerald-900/20">
              <div className="text-[11px] font-semibold tracking-wide text-emerald-700 uppercase dark:text-emerald-300">
                Verified {citations.length > 1 ? 'quotes' : 'quote'}
              </div>
              {citations.map((citation, index) => (
                <div key={index} className="font-serif italic">
                  “{citation.quote}”
                  {citation.match === 'approximate' && (
                    <span className="ml-1 font-sans text-[11px] text-ink-500 not-italic">(near-verbatim)</span>
                  )}
                </div>
              ))}
            </div>
          )}
          <div ref={passage} className="font-serif text-[14.5px] leading-relaxed whitespace-pre-wrap">
            <Highlighted text={source.text} quotes={citations.map((citation) => citation.quote)} />
          </div>
          <p className="mt-4 text-xs text-ink-500 dark:text-ink-400">
            This is the text the model was given as [{source.source_id}], extracted from the PDF. Tables appear as one row
            per line.
          </p>
        </div>
      ) : (
        <iframe key={`${source.document_id}-${page}`} src={pdfUrl(source.document_id, page)} className="flex-1" title="PDF" />
      )}
    </div>
  )
}
