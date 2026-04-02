import { useState, useRef, useEffect, Component, ReactNode } from 'react'
import { Send, Bot, BarChart3, Database, TrendingUp, Paperclip, Loader2, Trash2, LogIn, Download } from 'lucide-react'
import Markdown from 'markdown-to-jsx'
import { supabase } from './lib/supabase'
import LoginModal from './components/LoginModal'
import ConfirmModal from './components/ConfirmModal'
import { loadChatHistory, saveMessage, StoredMessage } from './lib/chatHistory'
import { Session, AuthChangeEvent } from '@supabase/supabase-js'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Array<string | { source: string; score?: number }>
}

// ── Error Boundary ────────────────────────────────────────────
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; error: string }> {
  constructor(props: any) {
    super(props)
    this.state = { hasError: false, error: '' }
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '16px', color: '#f87171', fontSize: '0.85rem', background: 'rgba(239,68,68,0.1)', borderRadius: '8px', border: '1px solid rgba(239,68,68,0.3)' }}>
          ⚠ Render error: {this.state.error}
        </div>
      )
    }
    return this.props.children
  }
}

// ── Safe Markdown renderer ────────────────────────────────────
function SafeMarkdown({ content }: { content: unknown }) {
  if (content === null || content === undefined) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  const text = typeof content === 'string' ? content : JSON.stringify(content)
  if (!text.trim()) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  return <Markdown>{text}</Markdown>
}

// ── Source label helper ───────────────────────────────────────
function getSourceLabel(src: string | { source: string; score?: number }): string {
  if (typeof src === 'string') return src
  if (src && typeof src === 'object' && 'source' in src) return src.source
  return ''
}

// ── App ───────────────────────────────────────────────────────
function App() {
  const [toast, setToast] = useState<{message: string, type: 'error' | 'success'} | null>(null)
  const [isInitializingAuth, setIsInitializingAuth] = useState(true)
  const [fileToDelete, setFileToDelete] = useState<string | null>(null)
  
  const showToast = (message: string, type: 'error' | 'success' = 'error') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 3500)
  }

  const [session, setSession] = useState<Session | null>(null)
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [activeView, setActiveView] = useState<'chat' | 'eval' | 'docs'>('chat')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: '👋 Welcome to **Nucleus**! Sign in to upload your documents and start asking questions. Click **Sign In** in the top-right corner to get started.' }
  ])
  const [loading, setLoading] = useState(false)
  const [evalData, setEvalData] = useState<any>(null)
  const [uploadStatus, setUploadStatus] = useState('')
  const [library, setLibrary] = useState<string[]>([])
  const [newQuestion, setNewQuestion] = useState('')

  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }: { data: { session: Session | null } }) => {
      setSession(session)
      setIsInitializingAuth(false)
    })
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session: Session | null) => {
      setSession(session)
      setIsInitializingAuth(false)
    })
    return () => subscription.unsubscribe()
  }, [])

  const fetchFiles = async () => {
    if (!session) return
    try {
      const res = await fetch('/api/v1/ingest/files', {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      })
      const data = await res.json()
      setLibrary(Array.isArray(data) ? data : [])
    } catch (e) { console.error(e) }
  }

  useEffect(() => { 
    if (session) {
      fetchFiles() 
      loadChatHistory().then((history: StoredMessage[]) => {
        if (history.length > 0) {
          setMessages(history as Message[])
        }
      })
    }
  }, [session])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return
    if (!session) { setShowLoginModal(true); return }
    const q = input.trim()
    setInput('')
    const userMsg: Message = { role: 'user', content: q }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    
    if (session) saveMessage(userMsg)

    try {
      const res = await fetch('/api/v1/chat/', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.access_token}`
        },
        body: JSON.stringify({ question: q })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const answer = typeof data.answer === 'string' && data.answer.trim()
        ? data.answer
        : 'No answer returned from the backend.'
      const assistantMsg: Message = {
        role: 'assistant',
        content: answer,
        sources: Array.isArray(data.sources) ? data.sources : []
      }
      setMessages(prev => [...prev, assistantMsg])
      
      if (session) saveMessage(assistantMsg)
    } catch (e: any) {
      const errMsg = `⚠ Error: ${e.message || 'Could not reach the AI backend.'}`
      setMessages(prev => [...prev, { role: 'assistant', content: errMsg }])
      showToast(errMsg, 'error')
    } finally {
      setLoading(false)
    }
  }

  const runEvaluation = async () => {
    if (!newQuestion.trim() || loading) return
    if (!session) { setShowLoginModal(true); return }
    setLoading(true)
    setEvalData(null)
    try {
      const res = await fetch('/api/v1/evaluate/', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.access_token}`
        },
        body: JSON.stringify({ questions: [newQuestion] })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setEvalData(await res.json())
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  const confirmDelete = async () => {
    if (!fileToDelete) return
    try {
      const res = await fetch(`/api/v1/ingest/files/${encodeURIComponent(fileToDelete)}`, { 
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${session?.access_token}` }
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      fetchFiles()
      showToast(`Deleted ${fileToDelete}`, 'success')
    } catch (e: any) {
      showToast(`Delete failed: ${e.message}`, 'error')
    }
    setFileToDelete(null)
  }

  const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return
    if (!session) { setShowLoginModal(true); e.target.value = ''; return }
    
    // Size check
    for (const file of files) {
      if (file.size > MAX_FILE_SIZE) {
        showToast(`File ${file.name} is larger than 10MB limit.`, 'error')
        e.target.value = ''; return
      }
    }

    let successCount = 0
    for (let i = 0; i < files.length; i++) {
        const file = files[i]
        setUploadStatus(`Indexing ${file.name} (${i + 1}/${files.length})…`)
        const fd = new FormData()
        fd.append('file', file)
        try {
          const res = await fetch('/api/v1/ingest/upload', { 
            method: 'POST', body: fd, headers: { 'Authorization': `Bearer ${session?.access_token}` }
          })
          if (!res.ok) {
              const d = await res.json()
              showToast(`Error: ${d.detail || res.statusText}`, 'error')
              continue
          }
          successCount++
          setMessages(prev => [...prev, { role: 'assistant', content: `✅ **Indexed:** ${file.name}` }])
        } catch(e) {
          showToast(`Upload failed for ${file.name}.`, 'error')
        }
    }
    setUploadStatus('')
    e.target.value = ''
    if (successCount > 0) {
        fetchFiles()
        showToast(`Successfully indexed ${successCount} file(s).`, 'success')
    }
  }

  const exportChat = () => {
    if (messages.length === 0) {
      showToast('No discussion to export.', 'error')
      return
    }
    const md = messages.map(m => `**${m.role.toUpperCase()}**\n\n${m.content}`).join('\n\n---\n\n')
    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chat-export-${new Date().toISOString().slice(0,10)}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    showToast('Discussion exported successfully!', 'success')
  }

  const go = (view: typeof activeView) => setActiveView(view)

  const pageTitle = activeView === 'chat' ? 'AI Research Session'
    : activeView === 'eval' ? 'System Audit'
    : 'Knowledge Library'

  const isGuest = !session

  if (isInitializingAuth) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#080b10' }}>
        <Loader2 className="animate-spin text-indigo-500" size={32} />
      </div>
    )
  }

  return (
    <>
    {showLoginModal && <LoginModal onClose={() => setShowLoginModal(false)} />}
    <ConfirmModal 
      isOpen={!!fileToDelete} 
      title="Delete Document" 
      message={`Are you sure you want to delete "${fileToDelete}"? This action cannot be undone.`}
      confirmText="Delete"
      onConfirm={confirmDelete}
      onCancel={() => setFileToDelete(null)}
    />
    {/* Global Toast */}
    {toast && (
      <div style={{
        position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)', zIndex: 9999,
        background: toast.type === 'error' ? '#ef4444' : '#10b981', color: 'white',
        padding: '10px 20px', borderRadius: '8px', fontSize: '0.85rem', fontWeight: 500,
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)', animation: 'fadeIn 0.2s ease-out'
      }}>
        {toast.message}
      </div>
    )}
    <div className="app-shell">

      {/* ── Desktop Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <TrendingUp size={18} color="white" />
          </div>
          <h2>Nucleus</h2>
        </div>

        <button className={`nav-link ${activeView === 'chat' ? 'active' : ''}`} onClick={() => go('chat')}>
          <Bot size={18} /> Session
        </button>
        <button className={`nav-link ${activeView === 'eval' ? 'active' : ''}`} onClick={() => go('eval')}>
          <BarChart3 size={18} /> Performance
        </button>
        <button className={`nav-link ${activeView === 'docs' ? 'active' : ''}`} onClick={() => go('docs')}>
          <Database size={18} /> Knowledge
        </button>

        {/* Sidebar bottom — auth-aware */}
        {isGuest ? (
          <div style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <button className="nav-link" style={{ color: 'var(--accent-text)', width: '100%' }}
              onClick={() => setShowLoginModal(true)}>
              <LogIn size={18} /> Sign In
            </button>
          </div>
        ) : (
          <div className="sidebar-user" style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '0 8px 12px' }}>
              <img
                src={session!.user.user_metadata.avatar_url || `https://ui-avatars.com/api/?name=${session!.user.email}`}
                alt="User"
                style={{ width: '32px', height: '32px', borderRadius: '50%', border: '2px solid rgba(99,102,241,0.4)' }}
              />
              <div style={{ minWidth: 0 }}>
                <div className="user-name" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {session!.user.user_metadata.full_name || 'User'}
                </div>
                <div className="user-email" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {session!.user.email}
                </div>
              </div>
            </div>
            <button className="nav-link" style={{ color: '#f87171', width: '100%' }}
              onClick={() => supabase.auth.signOut()}>
              Sign Out
            </button>
          </div>
        )}
      </aside>

      {/* ── Main ── */}
      <main className="main-content">
        <header className="view-header">
          <h1>{pageTitle}</h1>
          {uploadStatus && (
            <div className="upload-status">
              <Loader2 size={14} className="animate-spin" />
              {uploadStatus}
            </div>
          )}
          {/* Header auth & actions */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {activeView === 'chat' && !isGuest && messages.length > 1 && (
              <button 
                onClick={exportChat}
                style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(255,255,255,0.05)', color: 'white', border: '1px solid rgba(255,255,255,0.1)', padding: '6px 12px', borderRadius: '8px', fontSize: '0.8rem', cursor: 'pointer' }}
                title="Export Chat to Markdown"
              >
                <Download size={14} /> Export
              </button>
            )}
            {isGuest ? (
              <button className="header-login-btn" onClick={() => setShowLoginModal(true)}>
                <LogIn size={16} /> <span className="header-login-label">Sign In</span>
              </button>
            ) : (
              <img
                src={session!.user.user_metadata.avatar_url || `https://ui-avatars.com/api/?name=${session!.user.email}`}
                alt="avatar"
                title={session!.user.email}
                style={{ width: '32px', height: '32px', borderRadius: '50%', border: '2px solid rgba(99,102,241,0.35)', cursor: 'pointer' }}
                onClick={() => supabase.auth.signOut()}
              />
            )}
          </div>
        </header>

        <div className="scroll-area" ref={scrollRef}>

          {/* ── CHAT ── */}
          {activeView === 'chat' && (
            <div className="chat-container">
              {/* Guest banner */}
              {isGuest && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08))',
                  border: '1px solid rgba(99,102,241,0.25)',
                  borderRadius: 'var(--radius-lg)',
                  padding: '14px 18px',
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '12px',
                  flexWrap: 'wrap',
                }}>
                  <span style={{ fontSize: '0.88rem', color: 'var(--accent-text)' }}>
                    🔒 Sign in to upload documents and chat with your knowledge base.
                  </span>
                  <button className="header-login-btn" style={{ padding: '6px 14px', fontSize: '0.82rem' }}
                    onClick={() => setShowLoginModal(true)}>
                    Sign In
                  </button>
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`message-bubble ${m.role}`}>
                  <div className="bubble-content">
                    <ErrorBoundary>
                      <SafeMarkdown content={m.content} />
                    </ErrorBoundary>
                  </div>
                  {m.sources && m.sources.length > 0 && (
                    <div className="bubble-source">
                      Source: {getSourceLabel(m.sources[0])}
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="message-bubble assistant typing-indicator">
                  <div className="bubble-content">Thinking…</div>
                </div>
              )}
            </div>
          )}

          {/* ── DOCS ── */}
          {activeView === 'docs' && (
            <div className="view-container">
              <h2 className="section-title">Knowledge Base</h2>
              <p className="section-subtitle">
                {library.length} document{library.length !== 1 ? 's' : ''} indexed and ready.
              </p>
              <div className="card-grid">
                {library.length === 0 && (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    No documents yet. Upload via the 📎 button in Chat.
                  </p>
                )}
                {library.map((f, i) => (
                  <div key={i} className="premium-card doc-card">
                    <div className="doc-icon"><Database size={20} /></div>
                    <div className="doc-text">
                      <div className="doc-name">{f}</div>
                      <div className="doc-sub">Ready for querying</div>
                    </div>
                    <button
                      className="btn-delete"
                      onClick={() => setFileToDelete(f)}
                      title={`Delete ${f}`}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── EVAL ── */}
          {activeView === 'eval' && (
            <div className="view-container">
              <h2 className="section-title">System Audit</h2>
              <p className="section-subtitle">
                Run a Ragas-powered scientific benchmark against your RAG pipeline.
              </p>

              <div className="eval-input-bar">
                <input
                  className="eval-input"
                  type="text"
                  value={newQuestion}
                  onChange={e => setNewQuestion(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && runEvaluation()}
                  placeholder="Enter a test query to audit…"
                />
                <button className="btn-primary" onClick={runEvaluation} disabled={loading || !newQuestion.trim()}>
                  {loading ? 'Running…' : 'Execute Audit'}
                </button>
              </div>

              {evalData ? (
                <>
                  <div className="metrics-grid">
                    <div className="premium-card metric-card">
                      <div className="metric-label">Faithfulness</div>
                      <div className="metric-value" style={{ color: 'var(--accent-text)' }}>
                        {(evalData.summary?.faithfulness * 100 || 0).toFixed(0)}%
                      </div>
                    </div>
                    <div className="premium-card metric-card">
                      <div className="metric-label">Context Precision</div>
                      <div className="metric-value" style={{ color: 'var(--accent-blue)' }}>
                        {(evalData.summary?.context_precision * 100 || 0).toFixed(0)}%
                      </div>
                    </div>
                    <div className="premium-card metric-card">
                      <div className="metric-label">Answer Correctness</div>
                      <div className="metric-value" style={{ color: 'var(--accent-green)' }}>
                        {(evalData.summary?.answer_correctness * 100 || 0).toFixed(0)}%
                      </div>
                    </div>
                  </div>

                  <h4 style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Detailed Breakdown
                  </h4>

                  {(evalData.detailed_report || []).map((item: any, idx: number) => (
                    <div key={idx} className="audit-log-item">
                      <div className="audit-q">
                        <TrendingUp size={15} /> Q{idx + 1}: {item.question}
                      </div>
                      <div className="audit-answer"><strong>AI Answer:</strong> {item.answer}</div>
                      <div className="audit-gt">
                        <div className="audit-gt-label">Ground Truth (AI-generated)</div>
                        <div className="audit-gt-text">{item.ground_truth}</div>
                      </div>
                    </div>
                  ))}
                </>
              ) : (
                <div className="eval-empty">
                  Enter a question above and click <strong>Execute Audit</strong> to begin.
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Chat Footer ── */}
        {activeView === 'chat' && (
          <footer className="input-frame">
              <input
                type="file"
                multiple
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleUpload}
                accept=".pdf,.txt,.js,.ts,.py"
              />
            <div className="glass-input">
              <button
                className="btn-icon"
                onClick={() => fileInputRef.current?.click()}
                title="Upload document"
              >
                <Paperclip size={19} />
              </button>
              <input
                className="chat-field"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
                placeholder="Ask anything about your documents…"
              />
              <button className="btn-send" onClick={handleSend} disabled={loading}>
                <Send size={17} color="white" />
              </button>
            </div>
          </footer>
        )}

        {/* ── Mobile Bottom Nav ── */}
        <nav className="mobile-nav-bar">
          <button className={`mobile-nav-btn ${activeView === 'chat' ? 'active' : ''}`} onClick={() => go('chat')}>
            <Bot size={22} /> Chat
          </button>
          <button className={`mobile-nav-btn ${activeView === 'eval' ? 'active' : ''}`} onClick={() => go('eval')}>
            <BarChart3 size={22} /> Audit
          </button>
          <button className={`mobile-nav-btn ${activeView === 'docs' ? 'active' : ''}`} onClick={() => go('docs')}>
            <Database size={22} /> Library
          </button>
        </nav>

      </main>
    </div>
    </>
  )
}

export default App
