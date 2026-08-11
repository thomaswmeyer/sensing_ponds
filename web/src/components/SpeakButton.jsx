import { useEffect, useState } from 'react'
import { useI18n } from '../i18n/useI18n.jsx'
import { hasAudio, speak, stopSpeaking } from '../lib/speech.js'

/**
 * Plays a string aloud. Consistent placement and gesture across every screen --
 * a user relying on audio needs the control in the same place each time.
 *
 * Renders nothing when no audio is available for the locale. A control that
 * silently does nothing is worse than no control, especially for someone who
 * cannot read the label beside it.
 */
export default function SpeakButton({ id, size = 'md', label }) {
  const { locale, t } = useI18n()
  const [available, setAvailable] = useState(false)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    // Voice list populates asynchronously on most browsers, so re-check shortly
    // after mount rather than trusting the first synchronous read.
    setAvailable(hasAudio(locale, id))
    const timer = setTimeout(() => setAvailable(hasAudio(locale, id)), 400)
    return () => clearTimeout(timer)
  }, [locale, id])

  useEffect(() => () => stopSpeaking(), [])

  if (!available) return null

  const onClick = async (e) => {
    e.stopPropagation()
    if (playing) {
      stopSpeaking()
      setPlaying(false)
      return
    }
    setPlaying(true)
    const result = await speak(locale, id, t(id))
    if (result === 'unavailable') setAvailable(false)
    setTimeout(() => setPlaying(false), 1200)
  }

  return (
    <button
      type="button"
      className={`speak speak--${size} ${playing ? 'is-playing' : ''}`}
      onClick={onClick}
      aria-label={label ?? t('action.listen')}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path
          d="M11 5 6 9H2v6h4l5 4V5z"
          fill="currentColor"
        />
        <path
          d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    </button>
  )
}
