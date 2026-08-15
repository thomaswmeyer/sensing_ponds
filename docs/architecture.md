# Architecture: Field Capture & Identification App

A mobile web app that identifies floating aquatic plants on-device and contributes each observation to a growing ground-truth dataset.

Status: **design, not built.** Nothing in this document has been implemented. Decisions marked ⚠️ are ones I made without confirmation — see [Decisions taken](#decisions-taken-without-confirmation) before building.

Related: [classifier-options.md](classifier-options.md) · [datasets.md](datasets.md) · [backend-cloudflare.md](backend-cloudflare.md)

> **The server design below is superseded but not yet replaced.** Fastify +
> Postgres/PostGIS + S3 still describes the intent, but the free-Postgres expiry
> flagged in [Free Postgres expires after 30 days](#️-free-postgres-expires-after-30-days)
> rules that hosting out. [backend-cloudflare.md](backend-cloudflare.md) works the
> same requirements through Workers + R2 + D1 on the free tier, and revisits two
> decisions taken here: the presigned two-step upload, and PostGIS.
> No decision has been taken between them.

## What this is for

Two purposes, and the second is the one that compounds:

1. **Immediate:** a field user photographs a plant and learns what it is and what it can be used for.
2. **Strategic:** every capture becomes a geocoded, timestamped, human-validatable observation. Over a season this becomes a ground-truth dataset for retraining the classifier — and, if the satellite track is revived, for labelling imagery. See [Why the metadata matters more than the photo](#why-the-metadata-matters-more-than-the-photo).

The app is a data collection instrument that happens to be useful to the person holding it. Design accordingly: the capture path must never block on the network, and the metadata must be complete enough to remain useful long after the capture.

## System overview

```
┌─────────────────────── Mobile browser (PWA) ───────────────────────┐
│                                                                     │
│   Camera  ──►  Capture  ──►  ONNX Runtime Web (WASM/WebGPU)         │
│   getUserMedia   canvas         mobilenetv3_small · ~2 MB INT8      │
│                     │                        │                      │
│                     │                        ▼                      │
│                     │              Species + confidence             │
│                     │                        │                      │
│                     │                        ├──► Uses lookup       │
│                     │                        │    (static table)    │
│                     │                        │                      │
│                     ▼                        ▼                      │
│              Geolocation API          Result screen ──► Audio       │
│              (lat/lon/accuracy)       icon + photo     (pre-rec ta) │
│                     │                 (or "not sure")               │
│                     │                                               │
│                     ▼                                               │
│         IndexedDB outbox  ◄── survives offline / app close          │
│                     │                                               │
└─────────────────────┼───────────────────────────────────────────────┘
                      │  Background Sync (opportunistic upload)
                      ▼
┌──────────────────── Node.js API (Fastify) ─────────────────────────┐
│                                                                     │
│   POST /observations  ──►  validate ──►  S3 presigned PUT (image)   │
│                                    └──►  Postgres + PostGIS (meta)  │
│                                                                     │
│   Review UI  ──►  human labelling / validation queue                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Client

**PWA, not native.** ⚠️ Assumed from "Node.js web app". Trade-off stated plainly: a PWA gets you one codebase, no app-store friction, and instant updates. It costs you reliable background upload on iOS (Background Sync is Chromium-only), and EXIF/GPS handling is more limited. If field users are on iOS with intermittent connectivity, revisit this — it is the weakest assumption in the document.

### Inference: ONNX Runtime Web

The [training script](../src/train_mobile_classifier.py) already exports ONNX (`--export-tflite`), so this is the path of least resistance. Alternatives considered: TensorFlow.js (heavier runtime, extra conversion hop), and server-side inference (defeats the offline requirement and adds latency to the one interaction that must feel instant).

- **Model size** ~2 MB at INT8 — cache in the service worker on first load, then it is offline-forever.
- **Backend:** WebGPU where available, WASM+SIMD fallback. On a low-end Android, expect tens of milliseconds. You will not be compute-bound.
- **Preprocessing must match training exactly** — resize to 256, centre-crop 224, ImageNet mean/std normalisation. This is the single most common source of silent accuracy loss in browser deployment: a mismatched resize filter or a forgotten normalisation drops accuracy without any error. Write a fixture test comparing browser output to PyTorch output on the same image, tolerance ~1e-3.

### The abstain path is a product requirement, not a nicety

The [training script](../src/train_mobile_classifier.py) computes a confidence threshold that reaches a target precision, and reports when no threshold does. Wire that number into the client:

- Above threshold → show the species and its uses.
- Below threshold → **"Not sure — this doesn't look like one of the four species I know."** Still upload it. An uncertain observation is a *valuable* one: it is exactly what the human review queue should prioritise.

A four-class softmax will confidently classify a rock, a boot, or a fifth plant species. Shipping without an abstain path means confidently telling a field user that a photo of *Salvinia* is water hyacinth.

### Capture flow

1. `getUserMedia({ video: { facingMode: 'environment' } })` → live preview
2. On shutter: draw to `<canvas>`, keep two representations
   - **Full-resolution JPEG** (quality ~0.85) for upload and future re-labelling
   - **224×224 tensor** for inference
3. Request `navigator.geolocation.getCurrentPosition({ enableHighAccuracy: true })` **in parallel with inference** — GPS fix is the slow step (2–15 s cold), inference is milliseconds. Do not serialise these.
4. Show result as soon as inference completes; let location settle into the record afterwards.
5. Write the complete observation to the IndexedDB outbox.
6. Attempt upload; on failure it stays queued.

**Never block the result screen on network or GPS.** Field connectivity near water bodies is poor and users will be standing in the sun.

### Offline outbox

IndexedDB, holding the image blob and metadata. Records are removed only on server-confirmed receipt. Use a client-generated UUID as the idempotency key so retries cannot create duplicates.

Upload via Background Sync where supported; fall back to a retry-on-app-open with exponential backoff. ⚠️ Storage pressure is a real risk — full-resolution photos accumulate fast. Suggest capping the outbox (e.g. 200 observations) and surfacing a "N pending upload" indicator rather than silently dropping.

## Why the metadata matters more than the photo

An observation's long-term value is mostly in its metadata. A perfectly-focused photo with no location accuracy figure is far less useful than a mediocre photo with tight GPS and an extent estimate.

**Capture from day one — retrofitting is impossible:**

| Field | Why |
|---|---|
| `lat`, `lon` | Obviously |
| **`accuracy_m`** | **Critical.** Distinguishes an observation you can trust to a specific water body from one you cannot. Also the filter that decides which observations could ever serve as satellite labels. |
| `captured_at` (UTC + offset) | Phenology; seasonal domain shift; satellite revisit matching if that track resumes |
| **`mat_extent`** | Isolated plant / patch / large mat. Cheap for the user (a three-way prompt), and the difference between an observation that can anchor a coarse-resolution label and one that cannot. |
| `species_pred`, `confidence`, `model_version` | Lets you re-evaluate old observations when the model changes |
| `abstained` | Flags the review queue |
| `device`, `app_version` | Debugging domain shift |

`model_version` deserves emphasis: without it, a dataset accumulated across model revisions becomes uninterpretable — you cannot tell which predictions came from which model.

## Server

**Fastify** ⚠️ (over Express) — schema-based validation via JSON Schema is built in, which matters for an endpoint accepting geospatial data from the field. Express is a defensible alternative if the team knows it better; the design does not depend on the choice.

### `POST /observations`

Accepts metadata, returns a presigned S3 PUT URL for the image. **Two-step, deliberately:** the image never passes through the API process, so a slow upload on a bad connection does not hold a Node handler open.

- Validate `client_uuid` for idempotency — a retried upload must not duplicate a row.
- Reject implausible coordinates and timestamps (future dates, null-island).
- Rate-limit per device.
- Store the row with `upload_status = 'pending'`, flipped to `'complete'` by an S3 event notification.

### Storage

**Postgres + PostGIS.** ⚠️ Chosen because every query you will actually run is spatial — observations within a water body, near a Sentinel-2 tile, clustered by site. Doing this without PostGIS means reimplementing it badly.

```sql
CREATE TABLE observations (
  id             uuid PRIMARY KEY,
  client_uuid    uuid UNIQUE NOT NULL,        -- idempotency
  geom           geography(Point, 4326) NOT NULL,
  accuracy_m     real NOT NULL,
  captured_at    timestamptz NOT NULL,
  uploaded_at    timestamptz NOT NULL DEFAULT now(),
  image_key      text NOT NULL,               -- S3 object key
  upload_status  text NOT NULL DEFAULT 'pending',

  species_pred   text,
  confidence     real,
  abstained      boolean NOT NULL DEFAULT false,
  model_version  text NOT NULL,

  mat_extent     text,                        -- isolated | patch | large_mat
  device_info    jsonb,

  -- Human validation, populated later
  species_true   text,
  reviewed_by    text,
  reviewed_at    timestamptz,
  review_status  text NOT NULL DEFAULT 'unreviewed'
);

CREATE INDEX ON observations USING GIST (geom);
CREATE INDEX ON observations (review_status) WHERE review_status = 'unreviewed';
CREATE INDEX ON observations (captured_at);
```

Images to S3 (or R2/GCS) at `observations/{yyyy}/{mm}/{uuid}.jpg`. Keep originals at full resolution — future re-labelling and higher-resolution models depend on it, and storage is cheap relative to the cost of re-collecting field data.

## Deployment: Render

⚠️ Assumed target. The design survives Render largely intact, with **one hazard that will destroy data if unaddressed**.

### ⚠️ Free Postgres expires after 30 days

[Free Render Postgres databases expire 30 days after creation](https://render.com/docs/free), with a 14-day inaccessible grace period, and **support no backups of any kind**.

This is not a performance limit — it is total data loss on a fixed timer. For an app whose entire strategic purpose is accumulating field observations that cannot be re-collected, a free database is unusable beyond a throwaway demo.

**Go paid on Postgres before a single real observation is captured.** Basic-256mb is ~$6/mo. This is the cheapest insurance in the project: a season of field data is worth vastly more than the subscription, and it cannot be recovered once gone.

If a free database is used for early development, treat it as strictly disposable and never point a field user at it.

### PostGIS works

[Render Postgres supports PostGIS](https://render.com/docs/postgresql-extensions) via `CREATE EXTENSION postgis` on PostgreSQL 13+. The spatial schema in this document works as written — no redesign needed.

Caveats: the extension needs enabling manually via the dashboard's psql session, and some users report permission problems specifically on free instances. Another reason to start paid.

### ⚠️ The filesystem is ephemeral — object storage is mandatory

[All Render services have an ephemeral filesystem](https://render.com/docs/disks); free services cannot attach a persistent disk at all. Anything written locally is lost on every deploy, restart, or spin-down.

The design already routes images to S3-compatible object storage, so this changes nothing — but it removes the tempting shortcut. **Never write uploaded images to local disk on Render, even temporarily as a "we'll move it later" step.** Render has no first-party object storage; use Cloudflare R2 (no egress fees, S3-compatible) or AWS S3. The presigned-upload design means the image never transits the Render service at all, which also keeps you inside the 100 GB/month outbound bandwidth allowance.

### Cold starts affect uploads, not the app

Free web services [spin down after 15 minutes of inactivity, with 30–60 second cold starts](https://render.com/docs/free). Field usage will be intermittent and bursty, so nearly every upload session hits a cold start.

The offline-first design absorbs this well — capture and inference are fully client-side, so **a cold start is invisible to the user**; it only delays background upload. But set the client's fetch timeout above 60 seconds, or the outbox will treat a cold start as a failure and retry needlessly.

For production, a Starter service at ~$7/mo eliminates spin-down. Combined with Basic Postgres, running cost is roughly **$13/mo plus R2 storage** — which is the real figure worth planning around.

### Recommended configuration

| Component | Plan | Why |
|---|---|---|
| Web service (Fastify API) | Starter ~$7/mo | No spin-down; free tier's cold start is tolerable but not for production |
| Postgres + PostGIS | Basic-256mb ~$6/mo | **Non-negotiable.** Free expires at 30 days with no backups. |
| Object storage | Cloudflare R2 | No Render-native option; R2 avoids egress fees |
| Static PWA | Render Static Site (free) | Genuinely fine on free — no server state, and the CDN has no spin-down |

The PWA itself can stay on the free static tier indefinitely. The 30-day expiry applies to Postgres, not static hosting.

### Also worth setting up

- **Enable daily backups** on the paid database. Available on paid plans; not the default mindset.
- **Health check endpoint** so Render restarts a wedged process.
- **`render.yaml`** — infrastructure as code, so the deployment is reproducible rather than dashboard-configured.
- **Region**: choose the one nearest your users. For a Tamil Nadu / Sri Lanka audience, Singapore is the closest Render region — meaningfully better than the US default for upload latency on poor connections.

## Human validation loop

The point of the whole exercise. A minimal review UI:

1. **Queue** — unreviewed observations, prioritised: abstained first, then low-confidence, then a random sample of high-confidence ones (to catch confident errors, which are the dangerous kind).
2. **Review** — image, model prediction, map location. Reviewer confirms or corrects the species, or marks unusable.
3. **Export** — validated observations become a training set: `species_true` + image for the classifier, `geom` + `captured_at` + `mat_extent` for satellite labels.

**Do not retrain on unvalidated model predictions.** Self-training on your own outputs entrenches errors — the model becomes more confident about exactly the things it already gets wrong. Every training record must carry a human `species_true`.

Worth tracking: model-vs-human agreement over time, per species. A drop signals domain shift as capture spreads to new sites — which is precisely the signal you want.

## Uses lookup

Species → uses is a **static table shipped with the client**, not a model output. Classify, then index. Training a model to emit uses just memorises a table through a lossier channel.

**⚠️ Two safety conditions must be encoded in the table, not left to prose:**

- **Phytoremediation biomass concentrates heavy metals** — it must never be routed to fodder or food-crop compost. If the app suggests uses, this exclusion has to be explicit.
- **Hyacinth is a regulated invasive** in many jurisdictions where transporting live material is illegal. Suggesting "harvest it for handicrafts" without a jurisdiction caveat is advice that can put a user in legal trouble.

Recommend a `caution` field per use, rendered inline rather than buried in a disclaimer screen. See [classifier-options.md](classifier-options.md#species--uses-is-a-lookup-table-not-a-model-output) for sourced uses.

## Localisation and non-literate users

**Tamil is the primary language, not a translation target.** English is the second locale. Later extension to other Dravidian languages — Malayalam, Kannada, Telugu — is a stated goal, so structure for it now: the cost of designing for multiple scripts up front is near zero, and retrofitting is expensive.

This is the section most likely to be underestimated. A voice-and-icon interface for non-literate users is not a layer added to a text UI — it changes what the UI can be.

### Language selection

- Default from `navigator.language`, but **always show an explicit switcher on first run**, in-script (`தமிழ்` / `English`), never behind a settings menu. A user whose device is set to English may well want Tamil.
- Persist the choice locally; it is a device preference, not an account setting.
- Store the locale on each observation — useful for understanding who is contributing from where.

### Script rendering

Tamil is not Latin-with-different-glyphs. Practical consequences:

- **Ship the font; do not rely on the device.** Bundle a subsetted [Noto Sans Tamil](https://notofonts.github.io/noto-docs/specimen/NotoSansTamilUI/) WOFF2 in the service worker cache. Android device font coverage is inconsistent, and the failure mode is [tofu](https://symbolfyi.com/guides/tofu-missing-glyphs/) — empty boxes where the text should be. For an offline-first app this is not optional.
- ⚠️ **Subset carefully.** Aggressive subsetting is a known source of missing glyphs in Indic scripts, because required combining marks get stripped. Subset by the full Tamil Unicode block, not by observed character frequency.
- **Tamil text runs longer than English** — commonly 20–40% more. Fixed-width buttons that fit "Water hyacinth" will break on "ஆகாயத்தாமரை". Test layouts in Tamil first; if it fits in Tamil it fits in English, rarely the reverse.
- **Line breaking and clustering** differ. Avoid `text-overflow: ellipsis` on Tamil strings and never truncate mid-word.
- Use `lang="ta"` on the root element so the browser selects correct shaping and hyphenation.

### Speech synthesis

**⚠️ The Web Speech API is not dependable for Tamil, and its failure mode is silent.** Chrome on Android [returns an unfiltered list of languages rather than installed voices](https://developer.chrome.com/blog/web-apps-that-talk-introduction-to-the-speech-synthesis-api), and **if the Tamil voice pack is not installed it falls back to an English voice without error** — meaning the app reads Tamil text aloud in English phonetics to a user who cannot read the screen to notice. That is worse than silence.

Two-tier approach:

1. **Pre-recorded audio for all fixed strings** — species names, uses, cautions, navigation prompts. There are perhaps 60–100 such strings. Record them once with a native Tamil speaker, ship as compressed audio in the service worker cache. This is **the primary path**: it works fully offline, sounds natural, has no fallback risk, and gets the pronunciation of plant names right — which a generic TTS engine will not.
2. **Synthesised TTS only for genuinely dynamic content**, and only after verifying an actual Tamil voice exists:
   ```js
   const hasTamil = speechSynthesis.getVoices()
     .some(v => v.lang.startsWith('ta') && v.localService);
   ```
   Treat a missing voice as "no audio available", never as "use whatever voice is default".

Design the app so **almost nothing needs dynamic TTS.** If the entire interface is a fixed vocabulary, tier 1 covers it and tier 2 never ships.

#### Why device TTS cannot be the delivery mechanism

Researched 2026-08-12. Two corrections to what this document previously assumed:

- **The detection snippet above cannot be trusted on the platform that matters.**
  Chrome on Android lists `ta-IN` whether or not the voice pack is installed, so
  the check can return a voice that does not exist and then read Tamil in English
  phonetics. `localService` does not save it, and Safari reports `default: true`
  for every voice. There is no reliable client-side probe — synthesis exposes no
  audio buffer to inspect, and the substituted voice reports success. The only
  trustworthy signal is a human listening.
- **Tamil TTS is not present by default on phones sold in Tamil Nadu.** Google's
  engine treats Tamil as an on-demand download requiring connectivity, which the
  field does not have. Samsung's TTS engine — the default on many Galaxy devices
  — has no Tamil voice pack at all, so it fails even when online. India's IS 16333
  (Part 3) mandates Indic *display and input*, not speech, so no regulation
  guarantees a floor. Even once installed, Google's higher-quality Tamil voice is
  network-backed and silently degrades or fails offline.

**Piper has no Tamil voice** — the earlier reference here was wrong; `piper-voices`
covers bn/hi/ml/mr/ne/te/ur but not `ta`. Every genuinely bundleable Tamil model is
114 MB+ (MMS Tamil ONNX is 114 MB and CC-BY-NC, which also blocks commercial use),
against a current payload of ~19 MB. The only sub-10 MB option is eSpeak-ng, whose
robotic formant output is a poor fit for users who cannot fall back on reading.

So: **generate the tier-1 clips at build time rather than bundling a model or
trusting the device.** ~100 clips at 24 kbps mono Opus is well under 2 MB.
[AI4Bharat `vits_rasa_13`](https://huggingface.co/ai4bharat/vits_rasa_13) (CC-BY-4.0,
Tamil `TAM_F`, HF access is gated) runs on CPU at build time; `IndicF5` (MIT) and
`indic-parler-tts` (Apache-2.0) are larger but size is irrelevant offline. Sarvam's
Bulbul is a paid alternative that models Dravidian syllable timing, costing under
₹100 for the whole string set.

**A native Tamil speaker must review every generated clip before field use.** Indic
TTS reliably mispronounces exactly what this app depends on — plant names, local
place names, numerals — and a non-literate user has no way to detect the error.

### Designing for non-literate users

Audio alone does not make an interface accessible to someone who cannot read. The visual design has to carry meaning independently:

- **Every action has an icon and a photo, not just a label.** Species selection should show plant photographs — the strongest affordance available, and it is the thing the user is literally looking at.
- **A speaker icon on every screen** replaying that screen's audio. Consistent placement, always the same gesture.
- **No text-only error states.** "Upload failed" as a bare string is invisible to this user; use an icon plus audio plus colour.
- **Numbers and confidence are hard to convey.** Do not show "87% confident". Use three visual states — confident / uncertain / not sure — each with a distinct icon, colour, and audio phrase.
- **The abstain path especially needs audio.** "Not sure" is the message most likely to be missed, and the most important to land.
- **Test with actual non-literate users.** ⚠️ This cannot be validated from a desk, and it is the assumption in this document I have least ability to check.

### String management

- Standard i18n structure (`i18next` or similar), locale files keyed by string ID, **never string-keyed by English text** — that pattern makes adding Malayalam a rewrite.
- **Each string ID maps to both text and an audio asset.** Build the audio path into the schema from the start:
  ```json
  {
    "species.eichhornia.name": {
      "text": "ஆகாயத்தாமரை",
      "audio": "audio/ta/species.eichhornia.name.opus"
    }
  }
  ```
- **Uses and cautions must be translated, not just the UI.** The heavy-metal and invasive-species warnings are the highest-stakes strings in the app. Have them reviewed by a native speaker with domain knowledge — a mistranslated safety caution is worse than no caution.
- Keep a `translation_status` per locale so partially-translated languages can ship without silently falling back to English mid-screen.

### Bundle cost

Pre-recorded Tamil audio at ~100 strings, Opus-compressed, is roughly 1–3 MB — comparable to the model itself. Cache it in the service worker alongside the model on first load, and consider making the language pack a deliberate first-run download over Wi-Fi rather than something fetched in the field.

## Privacy

Geolocation is personal data, and this app records where a person stood, when.

- **Explicit consent** at first capture, explaining that location is uploaded and why.
- **Public review UIs must not expose precise coordinates.** Fuzz to ~1 km for any public view; keep precision in the private dataset.
- ⚠️ **GDPR applies if any user is in the EU** — you need a lawful basis, a retention period, and a deletion path. Worth settling before launch rather than after.
- Strip EXIF from uploaded images. It carries a second, uncontrolled copy of location plus device identifiers.

## Build order

Each stage is independently useful; none blocks on the next.

1. **Capture + on-device inference + uses table, in Tamil from the first commit.** No server. Proves the model works in a browser, that preprocessing matches, and that the Tamil layout holds. Highest technical risk, lowest infrastructure cost — do it first.
2. **Audio layer.** Pre-recorded Tamil for all fixed strings, speaker affordance on every screen. Needs a native speaker and a recording session, so start scheduling it during stage 1.
3. **Outbox + upload endpoint + storage.** Data starts accumulating. From here, every day of use grows the dataset.
4. **Review UI.** Only worth building once there is a queue to review.
5. **Export → retraining.** Closes the loop.

Stage 1 is where the project is most likely to fail (browser/PyTorch numerical mismatch), and it needs no infrastructure to discover that. Do not build the server first.

**Do not build in English and localise later.** An English-first build bakes in fixed-width layouts, text-only affordances, and string-keyed translations — all of which have to be undone. Building Tamil-first with English as the second locale costs almost nothing at the start and avoids a rewrite. The same argument applies to the audio layer: if the UI assumes text, adding voice later means redesigning screens, not adding files.

## Decisions taken without confirmation

Flagging these because they shape everything downstream and were not specified:

| Decision | Rationale | Revisit if |
|---|---|---|
| PWA, not native | One codebase, no app store, instant updates | iOS users with poor connectivity — Background Sync is Chromium-only |
| ONNX Runtime Web | Training script already exports ONNX | — |
| Postgres + PostGIS | Every real query is spatial | Never; this one is solid |
| Fastify | Built-in schema validation | Team knows Express better — low stakes |
| Presigned S3 upload | Keeps large uploads out of Node | Self-hosting without object storage |
| Full-resolution image retention | Future re-labelling and better models | Storage cost becomes material at scale |
| Pre-recorded audio over TTS | Web Speech API silently falls back to English when the Tamil voice pack is absent | Dynamic content proves unavoidable — then Piper/AI4Bharat, at tens of MB |
| Bundled Noto Sans Tamil | Device font coverage is inconsistent; failure mode is tofu | Never; this one is solid |
| Render, paid Postgres from day one | Free Postgres expires at 30 days with no backups — total loss of unrecoverable field data | Never run real observations on free Postgres |
| Cloudflare R2 for images | Render has no native object storage; ephemeral filesystem; R2 has no egress fees | Already on AWS — then S3 |

## Open questions

1. **iOS or Android primarily?** Decides whether the PWA choice holds.
2. **Who reviews?** Internal team, or crowd-sourced? Changes the review UI substantially — a trusted internal reviewer needs no consensus mechanism; a crowd does. **If reviewers are Tamil-speaking, the review UI needs localising too** — it is not an internal-English-only tool by default.
3. **Which jurisdiction(s)?** Determines the invasive-species legal caveats and whether GDPR applies. Tamil-speaking primary audience suggests Tamil Nadu and/or Sri Lanka — worth confirming, since it also affects which invasive-species regulations apply.
4. **Expected volume?** Dozens of observations a week is a different system from thousands a day. This design targets the former and will need a queue in front of the API for the latter.
5. **Is the four-class model the shipping model?** It only knows four species. In a region with a fifth common floating plant, the abstain rate will be high — which is correct behaviour, but a poor user experience worth planning for.
6. **Who records the Tamil audio, and who reviews the safety-caution translations?** Both need a native speaker; the cautions additionally need someone who understands the domain. This is a scheduling dependency for stage 2, not a coding task.
7. **What are the local Tamil names for the four species?** Use vernacular names, not transliterated Latin binomials. ⚠️ I have not verified the Tamil name used in the localisation example — have a native speaker confirm the regionally correct term before it ships.
8. **Are the target users non-literate, low-literate, or literate-in-Tamil?** These need materially different interfaces. The current design assumes it must work for all three, which is the safe assumption but not the cheapest.
