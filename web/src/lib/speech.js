/**
 * Audio playback for non-literate users.
 *
 * Two tiers, in strict priority order:
 *
 *   1. Pre-recorded audio. The primary path. Works offline, sounds natural, and
 *      gets plant-name pronunciation right -- which generic TTS does not.
 *   2. Speech synthesis, ONLY when a genuine local voice for the locale exists.
 *
 * Tier 2 is guarded carefully. Chrome on Android returns an unfiltered list of
 * languages rather than installed voices, and if the Tamil voice pack is absent
 * it falls back to an ENGLISH voice without raising an error. That would read
 * Tamil text aloud in English phonetics to a user who cannot read the screen to
 * notice -- worse than staying silent. So a missing voice means "no audio",
 * never "use the default voice".
 *
 * See docs/architecture.md#speech-synthesis.
 */

import { audioPath } from '../i18n/strings.js'

let cachedVoices = null

function loadVoices() {
  if (cachedVoices) return cachedVoices
  cachedVoices = window.speechSynthesis?.getVoices() ?? []
  return cachedVoices
}

// getVoices() is empty until the async voiceschanged event on most browsers.
if (typeof window !== 'undefined' && window.speechSynthesis) {
  window.speechSynthesis.addEventListener?.('voiceschanged', () => {
    cachedVoices = null
  })
}

/**
 * A local (on-device) voice for this locale, or null.
 *
 * `localService` is the critical filter: a remote voice needs network, which the
 * field does not have. A voice that only exists server-side is not usable here.
 */
export function findVoice(locale) {
  const voices = loadVoices()
  return (
    voices.find((v) => v.lang?.toLowerCase().startsWith(locale) && v.localService) ?? null
  )
}

export function canSynthesise(locale) {
  return findVoice(locale) !== null
}

let currentAudio = null

function stopAll() {
  if (currentAudio) {
    currentAudio.pause()
    currentAudio = null
  }
  window.speechSynthesis?.cancel()
}

/**
 * Speak a string ID. Resolves when playback starts, not when it finishes.
 *
 * Returns the tier used so callers can reflect it in the UI -- 'recorded',
 * 'synth', or 'unavailable'. A speaker button should look disabled when audio is
 * genuinely unavailable rather than appearing to work and doing nothing.
 */
export async function speak(locale, id, text) {
  stopAll()

  const src = audioPath(locale, id)
  if (src) {
    try {
      const audio = new Audio(src)
      currentAudio = audio
      await audio.play()
      return 'recorded'
    } catch {
      // Fall through to synthesis: a missing or unplayable file should not be
      // silent if a real voice happens to be available.
      currentAudio = null
    }
  }

  const voice = findVoice(locale)
  if (!voice || !text) return 'unavailable'

  const utterance = new SpeechSynthesisUtterance(text)
  utterance.voice = voice
  utterance.lang = voice.lang
  utterance.rate = 0.9 // slightly slow: field conditions are noisy
  window.speechSynthesis.speak(utterance)
  return 'synth'
}

export function stopSpeaking() {
  stopAll()
}

/**
 * Whether any audio at all is available for this ID.
 *
 * Used to decide whether to render a speaker button. Showing a control that
 * cannot work is worse than omitting it, especially for a user who is relying on
 * audio because they cannot read the label next to it.
 */
export function hasAudio(locale, id) {
  return audioPath(locale, id) !== null || canSynthesise(locale)
}
