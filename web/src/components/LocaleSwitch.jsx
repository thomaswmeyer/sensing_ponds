import { LOCALES, LOCALE_NAMES } from '../i18n/strings.js'
import { useI18n } from '../i18n/useI18n.jsx'

/**
 * Always visible, never behind a settings menu.
 *
 * Each option is written in its own script, so a user who cannot read the other
 * language can still find theirs.
 */
export default function LocaleSwitch() {
  const { locale, setLocale } = useI18n()

  return (
    <div className="locale-switch" role="group" aria-label="Language / மொழி">
      {LOCALES.map((code) => (
        <button
          key={code}
          type="button"
          lang={code}
          className={code === locale ? 'is-active' : ''}
          onClick={() => setLocale(code)}
          aria-pressed={code === locale}
        >
          {LOCALE_NAMES[code]}
        </button>
      ))}
    </div>
  )
}
