import { ArrowUp, Download, Library, Menu, Square, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamChat, type ChatEvent } from './api'
import { AssistantTurn } from './components/AssistantTurn'
import { PaperPicker } from './components/PaperPicker'
import { SourcePanel } from './components/SourcePanel'
import { Sidebar } from './components/Sidebar'
import { conversationToMarkdown, newId } from './lib/format'
import type { AssistantMessage, Citation, ConversationDetail, ConversationSummary, Health, Paper, Source, Turn } from './types'

const MAX_CHARS = 2000
const SCOPE_KEY = 'paper-desk.scope'
const SUGGESTIONS = [
  'What does SadTalker predict from the audio?',
  'How does Avatar Forcing keep latency low enough for real-time conversation?',
  'Which deepfake generation methods does DF40 cover?',
  'How is avatar fingerprinting different from deepfake detection?',
]

type Panel = { kind: 'papers' } | { kind: 'source'; turnId: string; source: Source; citations: Citation[] } | null

function readScope(): string[] {
  try {
    return JSON.parse(localStorage.getItem(SCOPE_KEY) ?? '[]')
  } catch {
    return []
  }
}

function turnsFrom(detail: ConversationDetail): Turn[] {
  const turns: Turn[] = []
  for (const message of detail.messages) {
    if (message.role === 'user') turns.push({ id: message.id, question: message.content })
    else if (turns.length && message.answer) turns[turns.length - 1].answer = message.answer
  }
  for (const turn of turns) if (!turn.answer) turn.error = 'This question was not answered.'
  return turns
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [papers, setPapers] = useState<Paper[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  const [scope, setScope] = useState<string[]>(readScope)
  const [panel, setPanel] = useState<Panel>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)

  const refreshConversations = useCallback(() => {
    api.conversations().then(setConversations).catch(() => undefined)
  }, [])

  useEffect(() => {
    api.papers().then(setPapers).catch(() => undefined)
    refreshConversations()
  }, [refreshConversations])

  // Poll health until the models are loaded.
  useEffect(() => {
    let timer: number | undefined
    const poll = () =>
      api
        .health()
        .then((value) => {
          setHealth(value)
          if (value.status === 'loading') timer = window.setTimeout(poll, 2000)
        })
        .catch(() => {
          timer = window.setTimeout(poll, 3000)
        })
    poll()
    return () => window.clearTimeout(timer)
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(SCOPE_KEY, JSON.stringify(scope))
    } catch {
      // storage unavailable; the scope just isn't remembered
    }
  }, [scope])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns])

  // Drop scope entries for papers that no longer exist.
  useEffect(() => {
    if (papers.length) setScope((current) => current.filter((id) => papers.some((paper) => paper.id === id)))
  }, [papers])

  const updateTurn = (id: string, patch: Partial<Turn>) =>
    setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)))

  const ask = async (question: string, paperIds: string[] = scope) => {
    const message = question.trim()
    if (!message || busy || health?.status !== 'ready') return
    const turnId = newId()
    setTurns((current) => [...current, { id: turnId, question: message }])
    setDraft('')
    setBusy(true)
    const controller = new AbortController()
    abort.current = controller
    let activeConversation = conversationId
    try {
      await streamChat(
        { message, conversation_id: conversationId, paper_ids: paperIds },
        (event: ChatEvent) => {
          switch (event.event) {
            case 'meta':
              if (!activeConversation) {
                activeConversation = event.data.conversation_id
                setConversationId(activeConversation)
                refreshConversations()
              }
              break
            case 'stage':
              updateTurn(turnId, { stage: event.data.stage })
              break
            case 'rewritten':
              updateTurn(turnId, { rewritten: event.data.standalone_question })
              break
            case 'sources':
              updateTurn(turnId, { sources: event.data.sources })
              break
            case 'answer':
              updateTurn(turnId, { answer: event.data })
              break
            case 'error':
              updateTurn(turnId, { error: `${event.data.message} (request ${event.data.request_id.slice(0, 8)})` })
              break
          }
        },
        controller.signal,
      )
    } catch (error) {
      const stopped = controller.signal.aborted
      updateTurn(turnId, { error: stopped ? 'Stopped.' : `Request failed: ${(error as Error).message}` })
    } finally {
      setBusy(false)
      abort.current = null
      refreshConversations()
      input.current?.focus()
    }
  }

  const openConversation = async (id: string) => {
    if (busy) return
    setSidebarOpen(false)
    setPanel(null)
    try {
      const detail = await api.conversation(id)
      setConversationId(id)
      setTurns(turnsFrom(detail))
    } catch {
      refreshConversations()
    }
  }

  const newConversation = () => {
    if (busy) return
    setConversationId(null)
    setTurns([])
    setPanel(null)
    setSidebarOpen(false)
    input.current?.focus()
  }

  const deleteConversation = async (id: string) => {
    if (!window.confirm('Delete this conversation?')) return
    await api.deleteConversation(id).catch(() => undefined)
    if (id === conversationId) newConversation()
    refreshConversations()
  }

  const feedback = async (answer: AssistantMessage, rating: 1 | -1) => {
    try {
      await api.feedback(answer.message_id, rating)
      setTurns((current) =>
        current.map((turn) =>
          turn.answer?.message_id === answer.message_id ? { ...turn, answer: { ...turn.answer, feedback: rating } } : turn,
        ),
      )
    } catch {
      // feedback is best-effort
    }
  }

  const exportConversation = () => {
    const title = conversations.find((item) => item.id === conversationId)?.title ?? 'Literature review notes'
    const blob = new Blob([conversationToMarkdown(title, turns)], { type: 'text/markdown' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${title.replace(/[^\w\- ]+/g, '').trim().slice(0, 60) || 'conversation'}.md`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  const ready = health?.status === 'ready'
  const activeTitle = conversations.find((item) => item.id === conversationId)?.title
  const scopeLabel =
    scope.length === 0
      ? `All ${papers.length || ''} papers`.replace('  ', ' ')
      : scope.length === 1
        ? (papers.find((paper) => paper.id === scope[0])?.title.split(':')[0] ?? '1 paper')
        : `${scope.length} papers`
  const placeholder = !health
    ? 'Connecting…'
    : health.status === 'loading'
      ? 'Loading models, this takes about a minute…'
      : health.status === 'error'
        ? 'The models failed to load; see the server log.'
        : 'Ask about the papers…'

  return (
    <div className="flex h-full">
      <Sidebar
        conversations={conversations}
        activeId={conversationId}
        health={health}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNew={newConversation}
        onSelect={openConversation}
        onDelete={deleteConversation}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-ink-200 px-3 py-2 dark:border-ink-700">
          <button className="rounded p-1.5 md:hidden" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
            <Menu size={18} />
          </button>
          <div className="min-w-0 flex-1 truncate text-sm font-medium">{activeTitle ?? 'New conversation'}</div>
          <button
            onClick={() => setPanel(panel?.kind === 'papers' ? null : { kind: 'papers' })}
            className={`flex max-w-[45%] items-center gap-1.5 rounded-lg border px-2.5 py-1 text-sm ${
              scope.length
                ? 'border-accent-500/50 bg-accent-50 text-accent-700 dark:bg-accent-700/30 dark:text-accent-100'
                : 'border-ink-300 dark:border-ink-600'
            }`}
            title="Choose which papers to search"
          >
            <Library size={15} className="shrink-0" />
            <span className="truncate">{scopeLabel}</span>
          </button>
          {turns.some((turn) => turn.answer) && (
            <button
              onClick={exportConversation}
              className="rounded-lg p-1.5 text-ink-500 hover:bg-ink-100 dark:hover:bg-ink-800"
              title="Export conversation as Markdown"
            >
              <Download size={16} />
            </button>
          )}
        </header>

        <div className="scrollbar-thin flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6">
            {turns.length === 0 ? (
              <div className="pt-[12vh]">
                <h1 className="font-serif text-3xl font-semibold">What would you like to know?</h1>
                <p className="mt-3 max-w-xl text-ink-600 dark:text-ink-300">
                  Ask about {papers.length || 'the'} papers on avatar generation, talking heads, deepfake detection and
                  avatar fingerprinting. Every answer cites the passages it relies on, each quote is checked against the
                  paper, and the assistant says so when the papers don't contain the answer.
                </p>
                <div className="mt-8 grid gap-2 sm:grid-cols-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => ask(suggestion)}
                      disabled={!ready}
                      className="rounded-xl border border-ink-200 bg-white px-4 py-3 text-left text-sm hover:border-accent-500/50 disabled:opacity-50 dark:border-ink-700 dark:bg-ink-800"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-8">
                {turns.map((turn) => (
                  <div key={turn.id} className="space-y-4">
                    <div className="flex justify-end">
                      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent-700 px-4 py-2.5 whitespace-pre-wrap text-ink-50">
                        {turn.question}
                      </div>
                    </div>
                    <AssistantTurn
                      turn={turn}
                      activeSource={panel?.kind === 'source' && panel.turnId === turn.id ? panel.source.source_id : null}
                      onOpenSource={(source, citations) => setPanel({ kind: 'source', turnId: turn.id, source, citations })}
                      onFeedback={feedback}
                    />
                  </div>
                ))}
                <div ref={bottom} />
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-ink-200 px-4 py-3 dark:border-ink-700">
          <form
            className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-ink-300 bg-white px-3 py-2 shadow-sm focus-within:border-accent-500 dark:border-ink-600 dark:bg-ink-800"
            onSubmit={(event) => {
              event.preventDefault()
              ask(draft)
            }}
          >
            <textarea
              ref={input}
              value={draft}
              onChange={(event) => setDraft(event.target.value.slice(0, MAX_CHARS))}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  ask(draft)
                }
              }}
              rows={1}
              placeholder={placeholder}
              disabled={!ready}
              className="max-h-48 min-h-[28px] flex-1 resize-none bg-transparent py-1 outline-none [field-sizing:content] disabled:cursor-not-allowed"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => abort.current?.abort()}
                className="rounded-full bg-ink-700 p-2 text-white dark:bg-ink-200 dark:text-ink-900"
                aria-label="Stop"
              >
                <Square size={14} />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!ready || !draft.trim()}
                className="rounded-full bg-accent-600 p-2 text-white disabled:bg-ink-300 dark:disabled:bg-ink-600"
                aria-label="Send"
              >
                <ArrowUp size={16} />
              </button>
            )}
          </form>
          <div className="mx-auto mt-1.5 flex max-w-3xl justify-between px-1 text-[11px] text-ink-400">
            <span>Answers come only from the papers and can still be wrong; check the cited passages.</span>
            {draft.length > MAX_CHARS - 200 && (
              <span>
                {draft.length}/{MAX_CHARS}
              </span>
            )}
          </div>
        </div>
      </main>

      {panel && (
        <aside className="fixed inset-0 z-40 flex flex-col bg-ink-50 lg:static lg:z-auto lg:w-[30rem] lg:border-l lg:border-ink-200 dark:bg-ink-900 lg:dark:border-ink-700">
          <div className="flex items-center justify-between border-b border-ink-200 px-4 py-2 dark:border-ink-700">
            <div className="text-sm font-semibold">
              {panel.kind === 'papers' ? 'Papers' : `Source ${panel.source.source_id}`}
            </div>
            <button onClick={() => setPanel(null)} className="rounded p-1 text-ink-500 hover:bg-ink-100 dark:hover:bg-ink-800" aria-label="Close panel">
              <X size={18} />
            </button>
          </div>
          <div className="min-h-0 flex-1">
            {panel.kind === 'papers' ? (
              <PaperPicker
                papers={papers}
                selected={scope}
                onChange={setScope}
                disabled={!ready || busy}
                onAsk={(question, paperId) => {
                  setScope([paperId])
                  setPanel(null)
                  ask(question, [paperId])
                }}
              />
            ) : (
              <SourcePanel
                key={`${panel.turnId}-${panel.source.source_id}`}
                source={panel.source}
                citations={panel.citations}
              />
            )}
          </div>
        </aside>
      )}
    </div>
  )
}
