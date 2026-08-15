import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { DEFAULT_LOCALE, LOCALES, STRINGS } from './strings.js'

// Do NOT rename to match the app. This is a localStorage key, not a label:
// renaming it discards every saved language preference and drops users back to
// the default. The 'pondy' spelling predates the rename to Sensing Ponds and
// stays deliberately.
const STORAGE_KEY = 'pondy.locale'
const I18nContext = createContext(null)

function initialLocale() {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved && LOCALES.includes(saved)) return saved

  // Device language is a hint, not an answer -- a user whose phone is set to
  // English may well want Tamil. The switcher is always visible, never buried
  // in a settings menu.
  const device = navigator.languages?.find((l) =>
    LOCALES.includes(l.split('-')[0]),
  )
  return device ? device.split('-')[0] : DEFAULT_LOCALE
}

export function I18nProvider({ children }) {
  const [locale, setLocaleState] = useState(initialLocale)

  const setLocale = useCallback((next) => {
    setLocaleState(next)
    localStorage.setItem(STORAGE_KEY, next)
  }, [])

  // Drives font shaping and line breaking. Tamil needs correct `lang` to render
  // and break properly.
  useEffect(() => {
    document.documentElement.lang = locale
  }, [locale])

  const value = useMemo(() => {
    const table = STRINGS[locale] ?? STRINGS[DEFAULT_LOCALE]
    const t = (id) => {
      const s = table[id]
      if (s !== undefined) return s
      // Fall back to the default locale rather than showing a raw key, but make
      // the gap visible in development so partial translations get caught.
      const fallback = STRINGS[DEFAULT_LOCALE][id]
      if (import.meta.env.DEV && fallback === undefined) {
        console.warn(`[i18n] missing string: ${id}`)
      }
      return fallback ?? id
    }
    return { locale, setLocale, t }
  }, [locale, setLocale])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used inside I18nProvider')
  return ctx
}
