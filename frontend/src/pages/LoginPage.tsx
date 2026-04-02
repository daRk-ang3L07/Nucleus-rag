import React, { useState } from 'react';
import { supabase } from '../lib/supabase';

const LoginPage: React.FC = () => {
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin },
    });
    if (error) {
      console.error('Error logging in:', error.message);
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#080b10',
      overflow: 'hidden',
      position: 'relative',
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>

      {/* Background glow orbs */}
      <div style={{
        position: 'absolute', top: '-10%', left: '-10%',
        width: '600px', height: '600px',
        background: 'radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%)',
        borderRadius: '50%', pointerEvents: 'none',
      }} />
      <div style={{
        position: 'absolute', bottom: '-15%', right: '-10%',
        width: '500px', height: '500px',
        background: 'radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)',
        borderRadius: '50%', pointerEvents: 'none',
      }} />
      <div style={{
        position: 'absolute', top: '40%', right: '10%',
        width: '200px', height: '200px',
        background: 'radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%)',
        borderRadius: '50%', pointerEvents: 'none',
      }} />

      {/* Card */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        width: '100%',
        maxWidth: '420px',
        margin: '24px',
        background: 'rgba(255,255,255,0.04)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '32px',
        padding: '48px 40px',
        textAlign: 'center',
        boxShadow: '0 25px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07)',
      }}>

        {/* Logo */}
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '28px' }}>
          <div style={{
            position: 'relative',
            width: '80px', height: '80px',
          }}>
            {/* Outer ring */}
            <div style={{
              position: 'absolute', inset: '-3px',
              borderRadius: '26px',
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6, #3b82f6)',
              padding: '2px',
            }}>
              <div style={{
                width: '100%', height: '100%',
                background: '#080b10',
                borderRadius: '24px',
              }} />
            </div>
            {/* Icon */}
            <div style={{
              position: 'absolute', inset: '0',
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              borderRadius: '22px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '28px', fontWeight: '800', color: 'white',
              letterSpacing: '-1px',
            }}>
              ✦
            </div>
          </div>
        </div>

        {/* Title */}
        <h1 style={{
          fontSize: '28px', fontWeight: '800',
          color: '#f1f5f9', margin: '0 0 8px',
          letterSpacing: '-0.5px',
        }}>
          Nucleus
        </h1>
        <p style={{
          fontSize: '14px', color: 'rgba(148,163,184,0.8)',
          margin: '0 0 36px', lineHeight: '1.6',
        }}>
          Your private, AI-powered knowledge base.<br />
          Upload documents. Ask anything.
        </p>

        {/* Feature Pills */}
        <div style={{
          display: 'flex', gap: '8px', justifyContent: 'center',
          flexWrap: 'wrap', marginBottom: '32px',
        }}>
          {['🔒 Private', '⚡ Instant', '🧠 AI-Powered'].map(tag => (
            <span key={tag} style={{
              padding: '4px 12px',
              background: 'rgba(99,102,241,0.12)',
              border: '1px solid rgba(99,102,241,0.25)',
              borderRadius: '999px',
              fontSize: '12px', color: '#818cf8',
              fontWeight: '500',
            }}>{tag}</span>
          ))}
        </div>

        {/* Google Button */}
        <button
          onClick={handleLogin}
          disabled={loading}
          style={{
            width: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            gap: '12px',
            padding: '16px 24px',
            background: loading ? 'rgba(255,255,255,0.06)' : 'white',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: '16px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '15px', fontWeight: '600',
            color: loading ? 'rgba(255,255,255,0.4)' : '#1f2937',
            transition: 'all 0.25s ease',
            boxShadow: '0 2px 12px rgba(0,0,0,0.15)',
          }}
          onMouseOver={e => {
            if (!loading) {
              (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(-2px)';
              (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 8px 24px rgba(0,0,0,0.2)';
            }
          }}
          onMouseOut={e => {
            (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
            (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 2px 12px rgba(0,0,0,0.15)';
          }}
        >
          {loading ? (
            <>
              <div style={{
                width: '20px', height: '20px',
                border: '2px solid rgba(255,255,255,0.2)',
                borderTopColor: '#6366f1',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }} />
              Redirecting…
            </>
          ) : (
            <>
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </>
          )}
        </button>

        {/* Footer */}
        <p style={{
          marginTop: '24px', fontSize: '11px',
          color: 'rgba(100,116,139,0.7)', lineHeight: '1.6',
        }}>
          By signing in, your data is encrypted and<br />
          only visible to you. Powered by Supabase Auth.
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
