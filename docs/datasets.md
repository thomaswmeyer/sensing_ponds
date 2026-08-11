# Datasets

Public image datasets relevant to water hyacinth and allied floating aquatic plants, as surveyed August 2026. Also see [classifier-options.md](classifier-options.md).

## Summary

| Dataset | Images | Annotation | Platform | Licence | Best for |
|---|---|---|---|---|---|
| [WaterHyacinth](#waterhyacinth) | 1,790 | Class label only | Smartphone, close-range | CC BY 4.0 | Mobile field-ID classifier |
| [AqUavplant](#aquavplant) | 197 (4K) | Binary + multiclass masks | UAV @ 2.5 m | CC BY 4.0 | Segmentation, coverage fraction |
| [iNaturalist / GBIF](#inaturalist--gbif) | Thousands | Class label (crowd) | Mixed | CC0 / CC-BY / CC-BY-NC | Geographic diversity |

No other hyacinth-specific public image dataset was found. These three plus own-capture is realistically the landscape.

## WaterHyacinth

**[Data in Brief, 2023](https://www.sciencedirect.com/science/article/pii/S2352340923009320)** · [Mendeley Data](https://data.mendeley.com/datasets/vz6z64nwby/1) · CC BY 4.0

⚠️ **Naming is misleading.** Despite the title, this is **not** four hyacinth species — it is four different genera of floating aquatic plants. This is more useful than the title suggests: it is exactly the hyacinth-vs-allied-species discrimination the remote sensing literature identifies as the hard problem.

| Class | Common name | Original images |
|---|---|---|
| *Eichhornia crassipes* | Water hyacinth | 450 |
| *Pistia stratiotes* | Water lettuce | 480 |
| *Lemna minor* | Common duckweed | 390 |
| *Monochoria korsakowii* | Heartleaf false pickerelweed | 450 |

- **Capture:** smartphone (Redmi Note 8 Pro / Note 11, Galaxy S10, iPhone 12), natural daylight, July–September 2023, Sirajganj and Pabna districts, Bangladesh
- **Resolution:** distributed resized to 224 × 224 RGB
- **Annotation:** class label per image only — **no boxes, no masks**. Folder-per-class layout, directly compatible with `torchvision.datasets.ImageFolder`
- **Structure:** 1,790 originals + 4,050 pre-augmented

**Use the 1,790 originals only.** The 4,050 augmented images are pre-baked rotations/noise/flips. Static augmentation is strictly worse than on-the-fly (identical variants every epoch), and augmented copies of a training image landing in the test split invalidate the metrics.

**Limitations:** two districts, three months, four camera models. Narrow domain — a random train/test split will report inflated accuracy because near-duplicate shots of the same pond on the same afternoon appear in both. See the split strategy in [classifier-options.md](classifier-options.md#mobile-field-identification-track).

## AqUavplant

**[Scientific Data, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661991/)** · Figshare + GitHub loader · CC BY 4.0

197 images, 3840 × 2160 (4K), 31 aquatic plant species, 9 sites in Bangladesh, October 2023 – January 2024. DJI Mavic 3 Pro.

**Why this matters more than the image count suggests:**

- **It has segmentation masks** — binary (plant vs. background) and multiclass (0–31), PNG. Coverage fraction is a segmentation output; WaterHyacinth cannot produce it at all.
- **It is drone imagery** — a genuine intermediate rung between phone photos and satellite, and the natural training set for a geocoded ground-truth → satellite transfer path.

All three key species present: *Eichhornia crassipes* (listed as "Kachuripana"), *Pistia stratiotes*, *Lemna minor* ("Duck Weeds").

**⚠️ Severely imbalanced.** Lily Pad appears in 123 of 197 images; Umbrella Plant and Duck Weeds appear in **one image each**. Unusable as a 31-class problem for rare classes. Use as a **binary or coarse 4-class problem** — hyacinth / other floating vegetation / open water / background.

**⚠️ Altitude caveat.** Flights were at **2.5 m** above water for a GSD of 0.04–0.05 cm/px. That is near-macro, not typical survey altitude. Do not assume transfer to a 50 m mapping flight.

## iNaturalist / GBIF

**[iNaturalist Research-grade Observations on GBIF](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7)** · CC0 / CC-BY / CC-BY-NC (mixed, filterable)

Not a curated ML dataset — a bulk download of citizen-science observations. *Eichhornia crassipes* is a globally tracked invasive with heavy observation coverage.

**This is the fix for WaterHyacinth's geographic narrowness**: global coverage, all seasons, every camera type, arbitrary framing.

- **Search both names.** The genus was reclassified — the accepted name is now ***Pontederia crassipes***, but older records use *Eichhornia crassipes*. Querying only one misses records.
- **Research-grade** = ≥2 identifiers with >2/3 agreement at species level or lower. Crowd consensus, not expert verification — expect some label noise.
- **Filter licences** to CC0/CC-BY if shipping a product. CC-BY-NC blocks commercial use.
- **Inconsistent framing** — close-ups, habitat shots, herbarium specimens, occasional misidentified lookalikes. Needs a cleaning pass.

Also worth pulling the same for *Pistia stratiotes*, *Lemna minor*, and *Salvinia molesta* as confuser classes.

## Gaps

- **No public satellite-resolution labelled hyacinth dataset** was found. Relevant only if the satellite track is revived; labels would have to be self-generated.
- **No mid-altitude (30–100 m) UAV dataset.** AqUavplant is 2.5 m. This is the gap own-capture should target if drone survey is planned.
- **Geocoded ground truth is the missing asset.** Capture GPS + accuracy estimate + timestamp + rough mat extent from day one. Retrofitting location onto existing photos is impossible, and the ground-to-satellite transfer path depends on label quality more than model choice.
