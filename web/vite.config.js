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
        // The runtime is the binding constraint at 12.86 MB raw -- Workbox
        // compares against the UNCOMPRESSED size, so this must clear 12.86 even
        // though only ~3.3 MB crosses the wire after brotli. Workbox drops
        // oversized files from the manifest with a build-log warning and no
        // error, so a cap set too low fails silently and only shows up as an app
        // that cannot classify offline.
        maximumFileSizeToCacheInBytes: 20 * 1024 * 1024,
        // The app must work in the field with no connectivity, so everything it
        // needs is precached on first load: the app shell, the model, the Tamil
        // fonts, the icons, and the recorded audio.
        // No png/webmanifest here either: the plugin adds the manifest and the
        // icons it references on its own, so globbing them duplicates them too.
        // The WASM runtime and its loader glue must be precached like everything
        // else. There is no separate single-threaded ORT binary to prefer: since
        // ~1.19, onnxruntime-web ships only `-threaded` builds, and
        // `ort-wasm-simd-threaded.wasm` (13 MB raw, ~3.3 MB brotli) IS the
        // single-threaded path -- `ort.env.wasm.numThreads = 1` in
        // src/lib/inference.js makes it run on one thread without
        // SharedArrayBuffer. Excluding it by name does not dodge a fatter build;
        // it just leaves the app unable to classify offline, which is the one
        // thing this app exists to do. Verify with a real airplane-mode install
        // before trusting any change here.
        globPatterns: ['**/*.{js,mjs,css,html,woff2,onnx,json,opus,mp3,wasm}'],
        globIgnores: [
          // The JSEP (WebGPU), asyncify and JSPI builds are 14-26 MB each and
          // nothing imports them: src/lib/inference.js uses the
          // `onnxruntime-web/wasm` entry point deliberately. If one of these ever
          // appears in dist/, that import has regressed -- fix the import rather
          // than widening this list.
          '**/ort-wasm-simd-threaded.jsep.*',
          '**/ort-wasm-simd-threaded.asyncify.*',
          '**/ort-wasm-simd-threaded.jspi.*',
        ],
      },
      manifest: {
        // Pin the app identity explicitly. Without `id`, the browser derives it
        // from start_url, so ever changing start_url would orphan every existing
        // install and reinstall as a separate app. Cheap to set now while there
        // are no installs; impossible to change later without breaking them.
        id: '/?app=pondy-plant-id',
        name: 'Pondy Plant ID',
        short_name: 'Plant ID',
        description: 'Identify floating water plants and record where they grow',
        lang: 'ta',
        start_url: '/',
        display: 'standalone',
        orientation: 'portrait',
        categories: ['education', 'utilities'],
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
        // Chrome on Android only shows the richer install dialog -- the one with
        // screenshots and the description, rather than a bare one-line bar --
        // when at least one screenshot declares form_factor 'narrow'. Google
        // measured install rates roughly doubling with it, which matters when a
        // field tester is deciding whether to trust an unfamiliar link.
        //
        // Constraints, all of which Chrome enforces silently by just not showing
        // the dialog: PNG or JPEG only, 320-3840px per side, longest side no more
        // than 2.3x the shortest, and every 'narrow' entry must share one aspect
        // ratio. Real device captures are the point -- see web/ASSETS-NEEDED.md.
        screenshots: [
          {
            src: 'screenshots/capture.png',
            sizes: '1080x2400',
            type: 'image/png',
            form_factor: 'narrow',
            label: 'Point the camera at a floating plant',
          },
          {
            src: 'screenshots/result.png',
            sizes: '1080x2400',
            type: 'image/png',
            form_factor: 'narrow',
            label: 'The plant is named in Tamil, with its uses',
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
