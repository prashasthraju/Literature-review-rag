import { BookOpen, MessageSquarePlus, Trash2, X } from 'lucide-react'
import type { ConversationSummary, Health } from '../types'
import { shortModel } from '../lib/format'

interface Props {
  conversations: ConversationSummary[]
  activeId: string | null
  health: Health | null
  open: boolean
  onClose: () => void
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

function relativeDate(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function Sidebar({ conversations, activeId, health, open, onClose, onNew, onSelect, onDelete }: Props) {
  const status = health?.status ?? 'loading'
  const dot = { loading: 'bg-amber-400 animate-pulse', ready: 'bg-emerald-500', error: 'bg-red-500' }[status]
  const label = {
    loading: 'Loading models…',
    ready: health ? shortModel(health.generator) : 'Ready',
    error: 'Models failed to load',
  }[status]

  return (
    <>
      {open && <div className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={onClose} />}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-ink-200 bg-ink-100 transition-transform md:static md:translate-x-0 dark:border-ink-700 dark:bg-ink-800 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center gap-2 px-4 pt-4 pb-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-700 text-ink-50">
            <BookOpen size={17} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="font-serif text-[17px] leading-tight font-semibold">Paper Desk</div>
            <div className="text-xs text-ink-500 dark:text-ink-400">Literature review assistant</div>
          </div>
          <button className="rounded p-1 text-ink-500 md:hidden" onClick={onClose} aria-label="Close sidebar">
            <X size={18} />
          </button>
        </div>

        <div className="px-3">
          <button
            onClick={onNew}
            className="flex w-full items-center gap-2 rounded-lg border border-ink-300 bg-white px-3 py-2 text-sm font-medium shadow-xs hover:bg-ink-50 dark:border-ink-600 dark:bg-ink-700 dark:hover:bg-ink-600"
          >
            <MessageSquarePlus size={16} /> New conversation
          </button>
        </div>

        <nav className="scrollbar-thin mt-4 flex-1 overflow-y-auto px-2 pb-2">
          {conversations.length > 0 && (
            <div className="px-2 pb-1 text-[11px] font-semibold tracking-wide text-ink-500 uppercase dark:text-ink-400">
              Conversations
            </div>
          )}
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`group flex items-center rounded-lg ${
                conversation.id === activeId ? 'bg-ink-200 dark:bg-ink-700' : 'hover:bg-ink-200/60 dark:hover:bg-ink-700/60'
              }`}
            >
              <button className="min-w-0 flex-1 px-2 py-2 text-left" onClick={() => onSelect(conversation.id)}>
                <div className="truncate text-sm">{conversation.title}</div>
                <div className="text-[11px] text-ink-500 dark:text-ink-400">{relativeDate(conversation.updated_at)}</div>
              </button>
              <button
                className="mr-1 rounded p-1.5 text-ink-400 opacity-0 group-hover:opacity-100 hover:text-red-600 focus:opacity-100"
                onClick={() => onDelete(conversation.id)}
                aria-label="Delete conversation"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </nav>

        <div className="border-t border-ink-200 px-4 py-3 text-xs text-ink-500 dark:border-ink-700 dark:text-ink-400">
          <div className="flex items-center gap-2" title={health?.detail ?? undefined}>
            <span className={`h-2 w-2 rounded-full ${dot}`} />
            <span className="truncate">{label}</span>
          </div>
          {health && (
            <div className="mt-1">
              {health.papers} papers · {health.retrieval_config} · {health.prompt_version}
            </div>
          )}
        </div>
      </aside>
    </>
  )
}
