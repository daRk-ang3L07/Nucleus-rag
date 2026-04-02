import { supabase } from './supabase'

export interface StoredMessage {
  id?: string
  role: 'user' | 'assistant'
  content: string
  sources?: Array<string | { source: string; score?: number }>
  created_at?: string
}

export async function loadChatHistory(): Promise<StoredMessage[]> {
  const { data, error } = await supabase
    .from('messages')
    .select('id, role, content, sources, created_at')
    .order('created_at', { ascending: true })
    .limit(100)

  if (error) {
    console.error('Failed to load history:', error.message)
    return []
  }

  return (data || []).map(row => ({
    id: row.id,
    role: row.role as 'user' | 'assistant',
    content: row.content,
    sources: row.sources ?? undefined,
    created_at: row.created_at,
  }))
}

export async function saveMessage(msg: StoredMessage): Promise<void> {
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return

  const { error } = await supabase.from('messages').insert({
    user_id: user.id,
    role: msg.role,
    content: msg.content,
    sources: msg.sources ?? null,
  })

  if (error) console.error('Failed to save message:', error.message)
}

export async function clearChatHistory(): Promise<void> {
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return

  const { error } = await supabase.from('messages').delete().eq('user_id', user.id)
  if (error) console.error('Failed to clear history:', error.message)
}
