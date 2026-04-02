import React, { useEffect } from 'react'

interface ConfirmModalProps {
  isOpen: boolean
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  onConfirm: () => void
  onCancel: () => void
}

const ConfirmModal: React.FC<ConfirmModalProps> = ({ 
  isOpen, title, message, confirmText = 'Confirm', cancelText = 'Cancel', onConfirm, onCancel 
}) => {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
      if (e.key === 'Enter') onConfirm()
    }
    if (isOpen) document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [isOpen, onCancel, onConfirm])

  if (!isOpen) return null

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-card" style={{ maxWidth: '360px', padding: '36px 28px 28px' }} onClick={e => e.stopPropagation()}>
        <h3 style={{ margin: '0 0 12px', fontSize: '1.25rem', fontWeight: 600, color: 'white' }}>{title}</h3>
        <p style={{ margin: '0 0 24px', fontSize: '0.9rem', color: 'rgba(255,255,255,0.7)', lineHeight: 1.5 }}>
          {message}
        </p>
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <button 
            onClick={onCancel}
            style={{ 
              padding: '10px 16px', background: 'transparent', border: '1px solid rgba(255,255,255,0.2)', 
              color: 'white', borderRadius: '8px', cursor: 'pointer', fontSize: '0.9rem' 
            }}>
            {cancelText}
          </button>
          <button 
            onClick={onConfirm}
            style={{ 
              padding: '10px 16px', background: '#ef4444', border: 'none', 
              color: 'white', borderRadius: '8px', cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500
            }}>
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ConfirmModal
