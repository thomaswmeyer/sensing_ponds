import { useCallback, useEffect, useRef, useState } from 'react'
import { useI18n } from '../i18n/useI18n.jsx'
import SpeakButton from './SpeakButton.jsx'

/**
 * Live camera preview and shutter.
 *
 * Capture produces two things from one frame: a full-resolution JPEG for upload
 * and future re-labelling, and the same canvas for inference. Keeping the
 * original at full resolution matters -- re-collecting field data is far more
 * expensive than storing pixels.
 */
export default function Camera({ onCapture }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const { t } = useI18n()
  const [state, setState] = useState('starting') // starting | ready | denied | error

  const start = useCallback(async () => {
    setState('starting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setState('ready')
    } catch (err) {
      setState(err?.name === 'NotAllowedError' ? 'denied' : 'error')
    }
  }, [])

  useEffect(() => {
    start()
    return () => streamRef.current?.getTracks().forEach((track) => track.stop())
  }, [start])

  const capture = useCallback(() => {
    const video = videoRef.current
    if (!video || state !== 'ready') return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    canvas.toBlob(
      (blob) => onCapture({ canvas, blob }),
      'image/jpeg',
      0.85,
    )
  }, [onCapture, state])

  if (state === 'denied' || state === 'error') {
    return (
      <div className="camera camera--blocked">
        <div className="blocked-card">
          <span className="blocked-icon" aria-hidden="true">📷</span>
          <h2>
            {t('camera.permission.title')}
            <SpeakButton id="camera.permission.title" size="sm" />
          </h2>
          <p>{t('camera.permission.body')}</p>
          <button type="button" className="btn btn--primary" onClick={start}>
            {t('camera.permission.retry')}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="camera">
      <video ref={videoRef} playsInline muted autoPlay className="camera__preview" />

      {state === 'starting' && <p className="camera__status">{t('camera.starting')}</p>}

      <button
        type="button"
        className="shutter"
        onClick={capture}
        disabled={state !== 'ready'}
        aria-label={t('action.capture')}
      >
        <span className="shutter__ring" />
      </button>
    </div>
  )
}
