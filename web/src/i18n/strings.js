/**
 * Localised strings, keyed by stable string ID.
 *
 * Keys are IDs, never English text. English-keyed translations make adding a
 * fourth locale a rewrite, and this project expects Malayalam / Kannada /
 * Telugu after Tamil.
 *
 * Every ID may carry an `audio` path alongside its text. Non-literate users are
 * a primary audience, so audio is part of the string contract rather than a
 * later addition -- see docs/architecture.md#designing-for-non-literate-users.
 *
 * TAMIL TEXT IS UNVERIFIED. It was written without a native speaker and must be
 * reviewed before any field use. The species names in particular are regional:
 * the term a Puducherry user recognises may differ from a Chennai one. The
 * safety cautions are the highest-stakes strings in the app -- a mistranslated
 * caution is worse than no caution.
 */

export const LOCALES = ['ta', 'en']
export const DEFAULT_LOCALE = 'ta'

export const LOCALE_NAMES = {
  ta: 'தமிழ்',
  en: 'English',
}

const ta = {
  'app.name': 'தாவரம் அறிக',
  'app.tagline': 'நீர்த் தாவரங்களை அடையாளம் காணுங்கள்',

  'action.capture': 'படம் எடுக்கவும்',
  'action.retake': 'மீண்டும் எடுக்கவும்',
  'action.listen': 'கேட்கவும்',
  'action.save': 'சேமிக்கவும்',
  'action.close': 'மூடவும்',

  'camera.permission.title': 'கேமரா அனுமதி தேவை',
  'camera.permission.body': 'தாவரத்தைப் படம் எடுக்க கேமரா அனுமதி வழங்கவும்.',
  'camera.permission.retry': 'மீண்டும் முயற்சிக்கவும்',
  'camera.starting': 'கேமரா தயாராகிறது…',

  'result.identifying': 'பரிசோதிக்கிறது…',
  'result.confident': 'இது:',
  'result.uncertain': 'இது இருக்கலாம்:',

  // The abstain path. This is the message most likely to be missed and the most
  // important to land -- it must be spoken, not only shown.
  'result.abstain.title': 'தெரியவில்லை',
  'result.abstain.body':
    'இந்தத் தாவரம் எனக்குத் தெரிந்த நான்கில் ஒன்றாகத் தெரியவில்லை. படம் சேமிக்கப்படும், ஒருவர் பின்னர் பார்ப்பார்.',

  'species.water_hyacinth': 'ஆகாயத்தாமரை',
  'species.water_lettuce': 'அகசத்தாமரை',
  'species.duckweed': 'ஆவாரை பாசி',
  'species.salvinia': 'ஆபிரிக்கப் பாசி',

  'uses.title': 'பயன்கள்',
  'uses.compost': 'உரம் தயாரிக்க',
  'uses.biogas': 'உயிர்வாயு தயாரிக்க',
  'uses.handicraft': 'கைவினைப் பொருட்கள்',
  'uses.fodder': 'கால்நடைத் தீவனம்',
  'uses.water_cleaning': 'நீரைச் சுத்தம் செய்ய',

  'caution.heavy_metals':
    'மாசடைந்த நீரிலிருந்து எடுத்தால், கால்நடைத் தீவனமாகவோ உணவுப் பயிர் உரமாகவோ பயன்படுத்த வேண்டாம்.',
  'caution.invasive':
    'இது ஒரு படையெடுப்புத் தாவரம். உயிருள்ள தாவரத்தை வேறு நீர்நிலைக்குக் கொண்டு செல்ல வேண்டாம்.',

  'location.acquiring': 'இருப்பிடம் தேடுகிறது…',
  'location.unavailable': 'இருப்பிடம் கிடைக்கவில்லை',
  'location.accuracy': 'துல்லியம்',

  'extent.question': 'எவ்வளவு தாவரம் உள்ளது?',
  'extent.isolated': 'ஒரு சில',
  'extent.patch': 'சிறு பரப்பு',
  'extent.large_mat': 'பெரும் பரப்பு',

  'sync.pending': 'அனுப்பப்படவில்லை',
  'sync.uploading': 'அனுப்புகிறது…',
  'sync.done': 'அனுப்பப்பட்டது',
  'sync.count': 'படங்கள் காத்திருக்கின்றன',

  'error.model': 'தாவரம் அறியும் திட்டம் ஏற்றப்படவில்லை',
  'error.generic': 'ஏதோ தவறு நடந்தது',
}

const en = {
  'app.name': 'Sensing Ponds',
  'app.tagline': 'Identify floating water plants',

  'action.capture': 'Take photo',
  'action.retake': 'Retake',
  'action.listen': 'Listen',
  'action.save': 'Save',
  'action.close': 'Close',

  'camera.permission.title': 'Camera access needed',
  'camera.permission.body': 'Allow camera access to photograph the plant.',
  'camera.permission.retry': 'Try again',
  'camera.starting': 'Starting camera…',

  'result.identifying': 'Identifying…',
  'result.confident': 'This is:',
  'result.uncertain': 'This might be:',

  'result.abstain.title': 'Not sure',
  'result.abstain.body':
    "This doesn't look like one of the four plants I know. The photo will be saved for someone to check later.",

  'species.water_hyacinth': 'Water hyacinth',
  'species.water_lettuce': 'Water lettuce',
  'species.duckweed': 'Duckweed',
  'species.salvinia': 'Salvinia',

  'uses.title': 'Uses',
  'uses.compost': 'Making compost',
  'uses.biogas': 'Making biogas',
  'uses.handicraft': 'Handicrafts',
  'uses.fodder': 'Animal fodder',
  'uses.water_cleaning': 'Cleaning water',

  'caution.heavy_metals':
    'If taken from polluted water, do not use as animal fodder or as compost for food crops.',
  'caution.invasive':
    'This is an invasive plant. Do not move live plants to another water body.',

  'location.acquiring': 'Finding location…',
  'location.unavailable': 'Location unavailable',
  'location.accuracy': 'Accuracy',

  'extent.question': 'How much plant is there?',
  'extent.isolated': 'A few plants',
  'extent.patch': 'A small patch',
  'extent.large_mat': 'A large mat',

  'sync.pending': 'Not sent yet',
  'sync.uploading': 'Sending…',
  'sync.done': 'Sent',
  'sync.count': 'photos waiting',

  'error.model': 'Could not load the plant identifier',
  'error.generic': 'Something went wrong',
}

export const STRINGS = { ta, en }

/**
 * Audio asset path for a string ID, or null if none is recorded.
 *
 * Pre-recorded audio is the primary voice path, not synthesised speech: Chrome
 * on Android silently falls back to an English voice when the Tamil voice pack
 * is absent, which would read Tamil text in English phonetics to a user who
 * cannot read the screen to notice. See docs/architecture.md#speech-synthesis.
 *
 * Returns null until the recording session happens; callers must handle that
 * and fall back to speech synthesis only after verifying a real local voice.
 */
export function audioPath(locale, id) {
  if (!RECORDED_AUDIO.has(`${locale}:${id}`)) return null
  return `/audio/${locale}/${id}.opus`
}

// Populated as recordings are made with a native speaker. Empty is correct for
// now -- an entry here without a matching file is a silent 404 in the field.
export const RECORDED_AUDIO = new Set()
