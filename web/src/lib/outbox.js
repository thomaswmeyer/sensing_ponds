/**
 * Offline-first observation queue.
 *
 * Field connectivity near water bodies is poor, so capture must never block on
 * the network. Observations are written to IndexedDB immediately and uploaded
 * opportunistically; records are removed only on server-confirmed receipt.
 *
 * The client-generated UUID is the idempotency key -- a retried upload after a
 * lost response must not create a duplicate row server-side.
 */

// Do NOT rename to match the app. This is the IndexedDB database name, not a
// label: renaming it points the app at a fresh, empty database and strands
// every queued observation in the old one. The outbox would read as empty while
// unsent captures sit in an abandoned database the app no longer opens -- the
// worst failure available to an offline-first queue, and a silent one. The
// 'pondy' spelling predates the rename to Sensing Ponds and stays deliberately.
const DB_NAME = 'pondy'
const DB_VERSION = 1
const STORE = 'outbox'

// Full-resolution photos accumulate quickly. Rather than silently dropping
// observations when storage fills, cap the queue and surface the count so the
// user can seek connectivity. Oldest-first eviction would lose the observations
// most likely to have been forgotten about, so we refuse new writes instead and
// tell the caller.
const MAX_PENDING = 200

export class OutboxFullError extends Error {}

let dbPromise = null

function openDb() {
  if (dbPromise) return dbPromise
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' })
        store.createIndex('status', 'status')
        store.createIndex('captured_at', 'captured_at')
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  return dbPromise
}

function tx(store, mode, fn) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const t = db.transaction(store, mode)
        const req = fn(t.objectStore(store))
        t.oncomplete = () => resolve(req?.result)
        t.onerror = () => reject(t.error)
        t.onabort = () => reject(t.error)
      }),
  )
}

export async function countPending() {
  const all = await tx(STORE, 'readonly', (s) => s.getAll())
  return all.filter((o) => o.status !== 'uploaded').length
}

/**
 * Queue an observation.
 *
 * `metadata` carries the fields that make an observation useful long after
 * capture -- see docs/architecture.md#why-the-metadata-matters-more-than-the-photo.
 * accuracy_m in particular cannot be reconstructed later and decides whether an
 * observation can ever anchor a coarse-resolution label.
 */
export async function enqueue({ blob, metadata }) {
  if ((await countPending()) >= MAX_PENDING) {
    throw new OutboxFullError(`outbox is full (${MAX_PENDING} pending uploads)`)
  }

  const record = {
    id: crypto.randomUUID(), // idempotency key, generated client-side
    status: 'pending',
    attempts: 0,
    queued_at: new Date().toISOString(),
    blob,
    ...metadata,
  }
  await tx(STORE, 'readwrite', (s) => s.add(record))
  return record.id
}

export async function listPending() {
  const all = await tx(STORE, 'readonly', (s) => s.getAll())
  return all.filter((o) => o.status !== 'uploaded')
}

export async function markUploaded(id) {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, 'readwrite')
    const store = t.objectStore(STORE)
    const get = store.get(id)
    get.onsuccess = () => {
      // Delete rather than flag: the blob is the bulk of the storage cost and
      // the server now holds it.
      if (get.result) store.delete(id)
    }
    t.oncomplete = resolve
    t.onerror = () => reject(t.error)
  })
}

export async function markFailed(id) {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, 'readwrite')
    const store = t.objectStore(STORE)
    const get = store.get(id)
    get.onsuccess = () => {
      const rec = get.result
      if (rec) store.put({ ...rec, attempts: rec.attempts + 1, status: 'pending' })
    }
    t.oncomplete = resolve
    t.onerror = () => reject(t.error)
  })
}
