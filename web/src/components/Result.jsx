import { useEffect } from 'react'
import { useI18n } from '../i18n/useI18n.jsx'
import { usesFor } from '../lib/uses.js'
import { speak } from '../lib/speech.js'
import SpeakButton from './SpeakButton.jsx'

/**
 * Identification result, or the abstain state.
 *
 * Confidence is never shown as a percentage -- "87% confident" is meaningless to
 * most users and unreadable to some. Three visual states instead, each with a
 * distinct icon, colour and spoken phrase.
 */
function confidenceBand(result) {
  if (result.abstain) return 'abstain'
  return result.confidence >= 0.85 ? 'confident' : 'uncertain'
}

export default function Result({ result, imageUrl, onRetake, extent, onExtentChange }) {
  const { locale, t } = useI18n()
  const band = confidenceBand(result)

  // Speak the headline automatically. For a non-literate user the result is the
  // whole point of the interaction, and the abstain message in particular is the
  // one most likely to be missed and the most important to land.
  useEffect(() => {
    const id = band === 'abstain' ? 'result.abstain.body' : `species.${result.label}`
    speak(locale, id, t(id))
  }, [band, locale, result.label, t])

  const { uses, speciesCautions } = band === 'abstain' ? { uses: [], speciesCautions: [] } : usesFor(result.label)

  return (
    <div className={`result result--${band}`}>
      <div className="result__photo">
        <img src={imageUrl} alt="" />
        {/* Retake belongs on the photo, not at the end of the page. The photo is
            what tells the user the shot was wrong -- blurred, too far, the wrong
            plant -- so the correction has to be reachable while that is on
            screen, rather than below the uses and the extent question where it
            needs a scroll the user has no reason to make. */}
        <button type="button" className="btn btn--ghost result__retake" onClick={onRetake}>
          {t('action.retake')}
        </button>
      </div>

      <div className="result__body">
        {band === 'abstain' ? (
          <section className="verdict verdict--abstain">
            <span className="verdict__icon" aria-hidden="true">?</span>
            <h2>
              {t('result.abstain.title')}
              <SpeakButton id="result.abstain.title" />
            </h2>
            <p>{t('result.abstain.body')}</p>
          </section>
        ) : (
          <section className="verdict">
            <p className="verdict__lead">
              {t(band === 'confident' ? 'result.confident' : 'result.uncertain')}
            </p>
            <h2>
              {t(`species.${result.label}`)}
              <SpeakButton id={`species.${result.label}`} />
            </h2>
            <ConfidenceMeter band={band} />
          </section>
        )}

        {speciesCautions.map((id) => (
          <Caution key={id} id={id} />
        ))}

        {uses.length > 0 && (
          <section className="uses">
            <h3>
              {t('uses.title')}
              <SpeakButton id="uses.title" size="sm" />
            </h3>
            <ul>
              {uses.map((use) => (
                <li key={use.id}>
                  <span className="use__label">
                    {t(use.id)}
                    <SpeakButton id={use.id} size="sm" />
                  </span>
                  {/* Cautions render inline with the use they qualify, never
                      collected into a disclaimer a user would skip or not hear. */}
                  {use.cautions.map((id) => (
                    <Caution key={id} id={id} inline />
                  ))}
                </li>
              ))}
            </ul>
          </section>
        )}

        <ExtentPicker value={extent} onChange={onExtentChange} />
      </div>
    </div>
  )
}

function ConfidenceMeter({ band }) {
  return (
    <div className={`meter meter--${band}`} aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}

function Caution({ id, inline }) {
  const { t } = useI18n()
  return (
    <p className={`caution ${inline ? 'caution--inline' : ''}`} role="note">
      <span className="caution__icon" aria-hidden="true">!</span>
      <span>{t(id)}</span>
      <SpeakButton id={id} size="sm" />
    </p>
  )
}

/**
 * Mat extent. Three choices, phrased as amounts rather than technical terms.
 *
 * Cheap for the user and impossible to reconstruct later -- it is the difference
 * between an observation that can anchor a coarse-resolution label and one that
 * cannot.
 */
function ExtentPicker({ value, onChange }) {
  const { t } = useI18n()
  const options = ['isolated', 'patch', 'large_mat']

  return (
    <section className="extent">
      <h3>
        {t('extent.question')}
        <SpeakButton id="extent.question" size="sm" />
      </h3>
      <div className="extent__options">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            className={value === opt ? 'is-active' : ''}
            onClick={() => onChange(opt)}
            aria-pressed={value === opt}
          >
            <span className={`extent__icon extent__icon--${opt}`} aria-hidden="true" />
            {t(`extent.${opt}`)}
          </button>
        ))}
      </div>
    </section>
  )
}
