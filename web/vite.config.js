import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    // HTTPS with a self-signed certificate. Not optional for phone testing:
    // getUserMedia and the service worker are both restricted to secure
    // contexts, and while localhost counts as secure, a LAN address does not.
    // Without this, iOS Safari silently refuses camera access.
    basicSsl(),
    VitePWA({
      registerType: 'prompt',
      // NB: no `includeAssets` here. It and globPatterns are independent inputs
      // to the same precache list and do NOT deduplicate -- listing the fonts,
      // model and config in both put every one of them in the manifest twice,
      // which meant re-downloading the 5.8 MB model a second time on install.
      // globPatterns below already covers everything, audio included.
      workbox: {
        // Default is 2 MB; the ONNX model and the WASM runtime both exceed it.
        maximumFileSizeToCacheInBytes: 12 * 1024 * 1024,
        // The app must work in the field with no connectivity, so everything it
        // needs is precached on first load: the app shell, the model, the Tamil
        // fonts, the icons, and the recorded audio.
        // No png/webmanifest here either: the plugin adds the manifest and the
        // icons it references on its own, so globbing them duplicates them too.
        globPatterns: ['**/*.{js,css,html,woff2,onnx,json,opus,mp3}'],
        globIgnores: [
          // The threaded/JSEP build is ~27 MB and needs cross-origin isolation
          // (COOP/COEP) to use its SharedArrayBuffer threads. We do not set those
          // headers, so it would download and then fall back to single-threaded
          // anyway -- 27 MB of a field user's data for nothing. The single-thread
          // SIMD build is a few hundred KB and is what actually runs.
          '**/ort-wasm-simd-threaded*.wasm',
          '**/ort-*.mjs',
        ],
      },
      manifest: {
        name: 'Pondy Plant ID',
        short_name: 'Plant ID',
        description: 'Identify floating water plants and record where they grow',
        lang: 'ta',
        start_url: '/',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#0b1f16',
        theme_color: '#0b1f16',
        icons: [
          { src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          {
            src: 'icons/icon-512-maskable.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
        ],
      },
      devOptions: { enabled: true, type: 'module' },
    }),
  ],
  // onnxruntime-web ships prebuilt WASM; bundling it breaks the worker loader.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    host: true, // reachable from a phone on the same network for real camera testing
  },
})
