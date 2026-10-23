import { ChevronDown, ChevronRight, FileText, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { pdfUrl } from '../api'
import type { Paper } from '../types'

interface Props {
  papers: Paper[]
  selected: string[]
  onChange: (ids: string[]) => void
  onAsk: (question: string, paperId: string) => void
  disabled: boolean
}

const QUICK_QUESTIONS: [string, (title: string) => string][] = [
  ['Contribution', (title) => `What problem does "${title}" address, and what is its main contribution?`],
  ['Method', (title) => `How does the method proposed in "${title}" work?`],
  ['Evaluation', (title) => `Which datasets and metrics does "${title}" use for evaluation, and what are the main results?`],
  ['Limitations', (title) => `What limitations or failure cases does "${title}" report?`],
]

export function PaperPicker({ papers, selected, onChange, onAsk, disabled }: Props) {
  const [filter, setFilter] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  const groups = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    const byTopic = new Map<string, Paper[]>()
    for (const paper of papers) {
      if (needle && !`${paper.title} ${paper.topic}`.toLowerCase().includes(needle)) continue
      byTopic.set(paper.topic, [...(byTopic.get(paper.topic) ?? []), paper])
    }
    return [...byTopic.entries()]
  }, [papers, filter])

  const toggle = (id: string) =>
    onChange(selected.includes(id) ? selected.filter((item) => item !== id) : [...selected, id])

  return (
    <div className="flex h-full flex-col">
      <div className="space-y-3 border-b border-ink-200 p-4 dark:border-ink-700">
        <p className="text-sm text-ink-600 dark:text-ink-300">
          {selected.length === 0
            ? `Answers draw on all ${papers.length} papers. Tick papers to search only those.`
            : `Searching ${selected.length} of ${papers.length} papers.`}
        </p>
        <div className="flex items-center gap-2">
          <label className="flex flex-1 items-center gap-2 rounded-lg border border-ink-300 bg-white px-2.5 py-1.5 dark:border-ink-600 dark:bg-ink-800">
            <Search size={14} className="text-ink-400" />
            <input
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter papers"
              className="w-full bg-transparent text-sm outline-none"
            />
          </label>
          {selected.length > 0 && (
            <button onClick={() => onChange([])} className="text-sm text-accent-600 hover:underline dark:text-accent-100">
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto p-2">
        {groups.map(([topic, items]) => (
          <section key={topic} className="mb-3">
            <h3 className="px-2 py-1 text-[11px] font-semibold tracking-wide text-ink-500 uppercase dark:text-ink-400">
              {topic}
            </h3>
            {items.map((paper) => {
              const isOpen = expanded === paper.id
              return (
                <div key={paper.id} className="rounded-lg hover:bg-ink-100 dark:hover:bg-ink-800">
                  <div className="flex items-start gap-2 px-2 py-2">
                    <input
                      type="checkbox"
                      checked={selected.includes(paper.id)}
                      onChange={() => toggle(paper.id)}
                      className="mt-1 accent-accent-600"
                      aria-label={`Search ${paper.title}`}
                    />
                    <button className="min-w-0 flex-1 text-left" onClick={() => setExpanded(isOpen ? null : paper.id)}>
                      <div className="text-sm leading-snug">{paper.title}</div>
                      <div className="text-[11px] text-ink-500 dark:text-ink-400">
                        {paper.pages} pages · {paper.chunks} passages
                      </div>
                    </button>
                    <button
                      className="mt-0.5 text-ink-400"
                      onClick={() => setExpanded(isOpen ? null : paper.id)}
                      aria-label="Show details"
                    >
                      {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </button>
                  </div>
                  {isOpen && (
                    <div className="space-y-2 px-8 pb-3">
                      <div className="flex flex-wrap gap-1.5">
                        {QUICK_QUESTIONS.map(([label, question]) => (
                          <button
                            key={label}
                            disabled={disabled}
                            onClick={() => onAsk(question(paper.title), paper.id)}
                            className="rounded-full border border-accent-500/40 px-2.5 py-0.5 text-xs text-accent-600 hover:bg-accent-50 disabled:opacity-40 dark:text-accent-100 dark:hover:bg-accent-700/40"
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      {paper.has_pdf && (
                        <a
                          href={pdfUrl(paper.id)}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-ink-500 hover:text-accent-600"
                        >
                          <FileText size={12} /> Open PDF
                        </a>
                      )}
                      {paper.sections.length > 0 && (
                        <ul className="space-y-0.5 border-l border-ink-200 pl-2 text-xs text-ink-500 dark:border-ink-700 dark:text-ink-400">
                          {paper.sections.slice(0, 12).map((section) => (
                            <li key={section} className="truncate">
                              {section}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </section>
        ))}
      </div>
    </div>
  )
}
