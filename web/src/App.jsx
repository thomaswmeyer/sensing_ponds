import { useCallback, useEffect, useRef, useState } from 'react'
import Camera from './components/Camera.jsx'
import Result from './components/Result.jsx'
import LocaleSwitch from './components/LocaleSwitch.jsx'
import SpeakButton from './components/SpeakButton.jsx'
import { useI18n } from './i18n/useI18n.jsx'
import { classify, loadModel } from './lib/inference.js'
import { countPending, enqueue, OutboxFullError } from './lib/outbox.js'

/**
 * Requests a position without ever blocking the result screen.
 *
 * A cold GPS fix takes 2-15 seconds; inference takes milliseconds. Serialising
 * them would make the app feel broken in exactly the conditions it is used in.
 * The fix is requested in parallel with capture and settles into the record
 * afterwards.
 */
function requestPosition() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) return resolve(null)
    navigator.geolocation.getCurrentPosition(
      (pos) =>
        resolve({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          // Required, not optional: without it there is no way to tell an
          // observation anchored to a specific water body from one that is not.
          accuracy_m: pos.coords.accuracy,
        }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 20000, maximumAge: 30000 },
    )
  })
}

export default function App() {
  const { t } = useI18n()
  const [phase, setPhase] = useState('camera') // camera | working | result
  const [result, setResult] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [extent, setExtent] = useState(null)
  const [pending, setPending] = useState(0)
  const [modelError, setModelError] = useState(false)

  const captureRef = useRef({ blob: null, position: null, capturedAt: null })

  // Warm the model at startup so the first capture is not the first download.
  useEffect(() => {
    loadModel().catch(() => setModelError(true))
    countPending().then(setPending)
  }, [])

  useEffect(() => () => imageUrl && URL.revokeObjectURL(imageUrl), [imageUrl])

  const onCapture = useCallback(async ({ canvas, blob }) => {
    setPhase('working')
    captureRef.current = {
      blob,
      capturedAt: new Date().toISOString(),
      position: null,
    }
    setImageUrl(URL.createObjectURL(blob))

    // Deliberately not awaited together: show the result the moment inference
    // finishes, and let the slower GPS fix land in the record behind it.
    const positionPromise = requestPosition()

    let prediction
    try {
      prediction = await classify(canvas)
    } catch {
      setModelError(true)
      setPhase('camera')
      return
    }

    setResult(prediction)
    setPhase('result')

    const position = await positionPromise
    captureRef.current.position = position

    try {
      await enqueue({
        blob: captureRef.current.blob,
        metadata: {
          captured_at: captureRef.current.capturedAt,
          ...(position ?? {}),
          species_pred: prediction.label,
          confidence: prediction.confidence,
          // An abstained observation is the most valuable one for the human
          // review queue, so it is uploaded like any other and flagged.
          abstained: prediction.abstain,
          mat_extent: null, // filled in below if the user answers
          locale: document.documentElement.lang,
        },
      })
      setPending(await countPending())
    } catch (err) {
      if (!(err instanceof OutboxFullError)) throw err
      // Surfaced through the pending badge rather than a blocking dialog --
      // losing the capture is worse than showing a full queue.
    }
  }, [])

  const onRetake = useCallback(() => {
    setPhase('camera')
    setResult(null)
    setExtent(null)
  }, [])

  return (
    <div className="app">
      <header className="app__header">
        <h1>
          {t('app.name')}
          <SpeakButton id="app.name" size="sm" />
        </h1>
        <LocaleSwitch />
      </header>

      {modelError && (
        <p className="banner banner--error" role="alert">
          <span aria-hidden="true">!</span> {t('error.model')}
          <SpeakButton id="error.model" size="sm" />
        </p>
      )}

      <main className="app__main">
        {phase === 'camera' && <Camera onCapture={onCapture} />}

        {phase === 'working' && (
          <div className="working">
            <span className="spinner" aria-hidden="true" />
            <p>{t('result.identifying')}</p>
          </div>
        )}

        {phase === 'result' && result && (
          <Result
            result={result}
            imageUrl={imageUrl}
            onRetake={onRetake}
            extent={extent}
            onExtentChange={setExtent}
          />
        )}
      </main>

      {pending > 0 && (
        <footer className="app__footer">
          <span className="pending-badge">{pending}</span>
          {t('sync.count')}
        </footer>
      )}
    </div>
  )
}
