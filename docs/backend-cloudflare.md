# Backend on Cloudflare: Workers + R2 + D1

Storage and upload for field observations, entirely on Cloudflare's free tier.

Status: **researched, not built. No decision taken.** Nothing here is implemented
and nothing has been provisioned. Two choices are deliberately left open — see
[Open decisions](#open-decisions). Figures verified against Cloudflare docs on
2026-08-15; free-tier terms change, so re-check before relying on them.

Related: [architecture.md](architecture.md) · [model-size.md](model-size.md)

## Why this replaces the backend in architecture.md

[architecture.md](architecture.md#server) specifies Fastify + Postgres/PostGIS +
S3, and then flags the problem that kills it: free Render Postgres
[expires 30 days after creation](https://render.com/docs/free) with no backups of
any kind. For an app whose entire strategic purpose is accumulating field
observations that cannot be re-collected, that is total data loss on a timer.

Cloudflare has no equivalent expiry, the client is already deployed there, and
the whole thing fits in the free tier with roughly two orders of magnitude of
headroom. The trade is losing PostGIS — addressed under
[Geospatial](#geospatial-postgis-is-the-real-loss).

## Fit against the free tier

Assuming 10–50 field testers and a few thousand photos per season:

| Product | Free allowance | Expected use | Headroom |
|---|---|---|---|
| Workers | 100k requests/day | a few hundred/day | ~1000× |
| R2 | 10 GB-month storage | 5–15 GB/season | ~1× ⚠️ |
| R2 ops | 1M Class A, 10M Class B /month | ~5k writes | ~200× |
| R2 egress | **free, unmetered** | — | n/a |
| D1 | 500 MB/database | a few hundred KB | ~1000× |
| D1 writes | 100k rows/day | a few hundred | ~200× |

R2's free egress is the load-bearing detail. The same design on S3 bills for
every photo review and every dataset export; here those are free forever.

Storage is the only line without comfortable headroom — see
[Cost](#cost-and-the-one-thing-that-can-actually-bill-you).

## Architecture

One Worker serving both the PWA's static assets and the API. Photos to R2,
metadata to D1.

```
Phone (offline-first)                    Cloudflare
─────────────────────                    ──────────
capture ──► IndexedDB outbox
                 │
                 │ when connectivity returns
                 ▼
         POST /api/observations ──────►  Worker
         multipart: meta + jpeg          │
                                         ├─► D1    INSERT OR IGNORE (uuid PK)
                                         └─► R2    put(obs/{uuid}.jpg, stream)
                 ◄───────────────────────┘
         200 {status} ──► delete local copy
```

The client half of this already exists.
[`web/src/lib/outbox.js`](../web/src/lib/outbox.js) generates a UUID per
observation, holds the blob in IndexedDB, and deletes only on server-confirmed
receipt. What is missing is the flush loop and the endpoint it talks to.

### Stream the image through the Worker; do not presign

[architecture.md](architecture.md#post-observations) specifies a two-step
presigned upload — metadata first, then the image direct to S3, so that a slow
upload never holds a Node handler open. **That reasoning does not transfer to
Workers, and following it here would make things worse.**

Pass the request body straight to R2 as a stream:

```js
await env.BUCKET.put(`obs/${uuid}.jpg`, request.body, {
  httpMetadata: { contentType: 'image/jpeg' },
})
```

- **CPU cost is near zero.** Waiting on network I/O does not count against the
  10 ms free-tier CPU limit, and a streamed body never lands in the isolate's
  128 MB memory. The limit that motivated presigning does not bind.
- **One request instead of three.** Presigning means request-URL → upload →
  confirm. On rural cellular with one bar, each extra round trip is another
  chance to fail — and this app is used standing next to a pond.
- **No orphan window.** The two-step design can land an image in R2 whose
  metadata never arrives, leaving objects to reconcile against nothing.
- **No SigV4, no bucket CORS, no S3 credentials.** Presigning needs real R2 API
  credentials in the signing path; a bucket binding needs none.

Free-plan request bodies cap at **100 MB** (an account-plan limit, not a Workers
one). Photos are 1–3 MB. Revisit presigning only if uploads approach that cap or
need to be resumable.

**Never `await request.arrayBuffer()`.** That buffers the whole photo into the
isolate, burns CPU against the 10 ms limit, and counts against the 128 MB memory
cap. This is the single easiest way to get this wrong.

### Idempotency, and the orphan trap in it

The client's contract — UUID generated on-device, local copy deleted only on a
2xx — means **retries are guaranteed**, so they have to be cheap and safe.

Make `uuid` the D1 PRIMARY KEY, insert with `INSERT OR IGNORE`, and branch on
`meta.changes === 0`. R2 `put()` to the same key is already idempotent
(last-write-wins, strongly consistent), so a duplicate photo upload is harmless;
D1 is where a duplicate would actually corrupt the dataset.

⚠️ **The failure this introduces:** if the D1 insert succeeds and the R2 put then
fails, the client retries, `INSERT OR IGNORE` reports a duplicate, the handler
returns 200 — and the photo is orphaned permanently, with a metadata row
pointing at an object that does not exist. The client deletes its only copy.

So the duplicate branch **must not** be a bare early return:

```js
if (inserted.meta.changes === 0) {
  // Already have the row. Confirm we also have the image -- a crash between
  // the insert and the put would otherwise strand this row forever, and the
  // client is about to delete its only copy.
  const existing = await env.BUCKET.head(key)
  if (existing) return json({ status: 'duplicate' })
  // fall through and re-put
}
```

R2 limits concurrent writes to a single object to 1/second (429 above that).
Irrelevant across distinct UUIDs, but a reason to keep client retry backoff.

### Schema

D1 is SQLite. The fields are those
[architecture.md](architecture.md#why-the-metadata-matters-more-than-the-photo)
identifies as impossible to retrofit.

```sql
CREATE TABLE observations (
  uuid           TEXT PRIMARY KEY,      -- client-generated; the idempotency key
  captured_at    TEXT NOT NULL,         -- ISO 8601 UTC
  received_at    TEXT NOT NULL,
  lat            REAL,                  -- null until COLLECT_POSITION is on
  lon            REAL,
  accuracy_m     REAL,                  -- decides if this can ever anchor a label
  species_pred   TEXT,
  confidence     REAL,
  abstained      INTEGER NOT NULL,      -- 0/1; abstains are the review priority
  mat_extent     TEXT,
  locale         TEXT,
  model_version  TEXT,                  -- lets old observations be re-evaluated
  image_key      TEXT NOT NULL,         -- R2 object key
  device_token   TEXT,                  -- provenance; see Auth
  review_status  TEXT DEFAULT 'unreviewed'
);

CREATE INDEX idx_obs_captured   ON observations(captured_at);
CREATE INDEX idx_obs_unreviewed ON observations(review_status)
  WHERE review_status = 'unreviewed';
CREATE INDEX idx_obs_bbox       ON observations(lat, lon);
```

Photos at `obs/{yyyy}/{mm}/{uuid}.jpg`. Keep originals at full resolution —
re-labelling and higher-resolution models both depend on it, and storage is cheap
against the cost of re-collecting field data.

### Geospatial: PostGIS is the real loss

D1 has no SpatiaLite, no R-tree, no geography types. Every spatial query in
[architecture.md](architecture.md#storage) has to be rewritten as a bounding box
plus Haversine in JS:

```sql
SELECT * FROM observations WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
```

At a few thousand rows this is genuinely fine — a full scan of 3,000 rows costs
3,000 of 5,000,000 daily row reads, and the Haversine filter afterwards is
microseconds. It stops being fine at maybe 10⁵–10⁶ rows or if real spatial joins
against water-body polygons become routine.

That is a plausible future for this project, not a fanciful one. The mitigation
is that D1 is SQLite and exports cleanly: if the analysis outgrows it, move the
metadata to Postgres/PostGIS and leave the photos in R2 untouched. **The schema
above is deliberately portable — no Cloudflare-specific types.**

## Auth

Threat model: an unadvertised endpoint on a small research project. The realistic
risks are opportunistic scanners and a lost or ex-tester's phone. There is no
adversary motivated to attack an aquatic-plant dataset. The real cost of
over-engineering is a field user unable to submit while standing in a marsh.

**Per-device bearer token**, checked against a `devices` table in D1.

Extractable from the client bundle, so it is a bot filter and not a security
boundary — but it makes the endpoint invisible to untargeted scanning, which is
the actual threat. Over a single shared secret it adds: revoking one lost phone
without redeploying to all 50, and provenance on every row.

Server-side limits matter more than the token, because they bound
[what a leaked token can cost](#cost-and-the-one-thing-that-can-actually-bill-you):

- reject bodies over ~10 MB
- require `Content-Type: image/jpeg`
- sanity-check lat/lon ranges and timestamp bounds
- optionally one WAF rate-limiting rule (one is free) on the upload path

**Rejected for the capture path:**

- **Turnstile** — tokens are short-lived and these uploads sit queued offline for
  hours. An observation captured in the field would arrive with an expired token.
- **Cloudflare Access** — requires authenticating against an identity provider,
  which is exactly what shared and borrowed phones with no accounts cannot do.

**Access is right for the review/export interface**, though: a handful of named
researchers, free under 50 seats, and that surface exposes the entire dataset
rather than accepting one row.

## Review and export

A second authenticated route reading from D1 (`GET /api/export?format=csv`), with
images served through a Worker route proxying `env.BUCKET.get()`.

⚠️ **Do not expose the bucket through its `r2.dev` public subdomain.** Cloudflare
documents it as testing-only and rate-limits it. Use a custom domain or
Worker-proxied reads.

## Cost, and the one thing that can actually bill you

**Workers, D1 and KV hard-stop at the free limit** — operations fail with an
error. No surprise bill.

**R2 does not. It bills.** There is no free-tier spending cap.

At realistic volumes this is pennies:

| Stored | Monthly |
|---|---|
| 10 GB | $0.00 (at the free line) |
| 15 GB | ~$0.08 |
| 50 GB (several seasons) | ~$0.60 |

Storage past 10 GB is $0.015/GB-month; egress and reads stay free. The exposure
is not the expected case but the pathological one — a leaked token and a scripted
uploader. Hence the size caps above, plus a billing alert.

⚠️ **Unverified: R2 requires completing a checkout flow even for free-tier use,
and the docs do not state whether a payment method is mandatory to complete it.**
For a project that may not have a card to put down, confirm this in the dashboard
before committing to the stack. This is the one item that could invalidate the
whole approach and it could not be settled from documentation.

## Open decisions

### 1. Migrate to Workers, or add Functions to Pages?

The app currently deploys to Pages via direct upload.

**Migrating to Workers** is the recommendation. Cloudflare's stated position is
that Pages remains supported but all new investment goes to Workers, and a single
Worker with `assets: { directory: "./dist" }` serves the PWA *and* `/api/*` from
one deploy, with bindings in `wrangler.jsonc`. Static asset requests stay free and
do not count against the 100k/day. Adding a backend is the natural moment.

**Staying on Pages Functions** works too, but carries a trap: bindings only apply
once `pages_build_output_dir` is added to the Wrangler file, and adding it
silently promotes whatever local dev bindings are configured to production. Use
`[env.production]` / `[env.preview]` sections if going this route.

Cost of migrating: the deploy command changes (`wrangler deploy`, not
`wrangler pages deploy`), and `_headers` becomes Workers static-asset config.

### 2. Downscale photos client-side?

Currently [`Camera.jsx`](../web/src/components/Camera.jsx) keeps a
full-resolution JPEG for upload.

**Downscaling to ~1 MB** puts 5,000 photos at ~5 GB — under the free line
indefinitely — and materially shortens uploads on bad connections, which is a
field-usability win as much as a cost one.

**Against:** full-resolution originals are what make future re-labelling and
higher-resolution models possible, and this data cannot be re-collected. A
compromise is downscaling confident predictions and keeping full resolution for
abstains, which are the observations the review queue actually cares about.

## Build order

Roughly a day, in dependency order:

1. Provision R2 bucket + D1 database; apply schema (**verify the R2 checkout
   question first** — it gates everything else)
2. Worker: `POST /api/observations`, with the duplicate-branch `head()` check
3. Flush loop in the client — retry with backoff, `markUploaded` on 2xx,
   `markFailed` otherwise. `outbox.js` already exposes all of it.
4. Flip `COLLECT_POSITION` to `true` in [`App.jsx`](../web/src/App.jsx) —
   coordinates finally have somewhere to go, which is the condition that comment
   names
5. Export route behind Access

Steps 2–3 are the only real work; the client contract already dictates their
shape.
