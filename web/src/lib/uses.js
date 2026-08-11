/**
 * Species -> uses. A static lookup table, deliberately not a model output.
 *
 * Training a network to emit uses would just memorise this table through a
 * lossier channel. Classify the species, then index here.
 *
 * The `cautions` are the highest-stakes content in the app and must render
 * inline with the use they qualify -- never collected into a disclaimer screen
 * that a non-literate user will not hear. Two conditions matter:
 *
 *   - Phytoremediation biomass concentrates heavy metals. It must not be routed
 *     to fodder or food-crop compost.
 *   - Water hyacinth is a regulated invasive in many jurisdictions, where moving
 *     live material is illegal.
 *
 * A naive species->uses mapping gets both wrong in ways that can harm someone.
 *
 * Sources for the hyacinth entries are cited in
 * docs/classifier-options.md#species--uses-is-a-lookup-table-not-a-model-output.
 */

export const USES = {
  water_hyacinth: {
    uses: [
      { id: 'uses.compost', cautions: ['caution.heavy_metals'] },
      { id: 'uses.biogas', cautions: [] },
      { id: 'uses.handicraft', cautions: [] },
      { id: 'uses.fodder', cautions: ['caution.heavy_metals'] },
      { id: 'uses.water_cleaning', cautions: [] },
    ],
    // Applies to the species as a whole, not to one use.
    speciesCautions: ['caution.invasive'],
  },

  water_lettuce: {
    uses: [
      { id: 'uses.compost', cautions: ['caution.heavy_metals'] },
      { id: 'uses.biogas', cautions: [] },
      { id: 'uses.water_cleaning', cautions: [] },
    ],
    speciesCautions: ['caution.invasive'],
  },

  duckweed: {
    uses: [
      { id: 'uses.fodder', cautions: ['caution.heavy_metals'] },
      { id: 'uses.compost', cautions: ['caution.heavy_metals'] },
      { id: 'uses.water_cleaning', cautions: [] },
    ],
    speciesCautions: [],
  },

  // Salvinia molesta is one of the world's most damaging aquatic weeds, and its
  // documented uses are thin. Listing weak uses would imply an encouragement to
  // harvest and move it that the evidence does not support -- so the entry
  // carries the invasive caution and nothing else.
  salvinia: {
    uses: [],
    speciesCautions: ['caution.invasive'],
  },
}

export function usesFor(species) {
  return USES[species] ?? { uses: [], speciesCautions: [] }
}
