/// <reference types="vite/client" />
import { createClient } from '@supabase/supabase-js'

declare global {
  interface Window {
    VITE_SUPABASE_URL?: string;
    VITE_SUPABASE_ANON_KEY?: string;
  }
}

// Support both build-time and runtime environment variables
const supabaseUrl = window.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = window.VITE_SUPABASE_ANON_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('Missing Supabase variables at build-time. Falling back to runtime config...')
}

export const supabase = createClient(supabaseUrl || '', supabaseAnonKey || '')
