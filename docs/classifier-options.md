# Classifier Options for Water Hyacinth & Water Pollution Detection

Status: scoping notes, validated against published literature (August 2026). No imagery source or ground truth confirmed yet — see [Open Questions](#open-questions). Claims sourced to papers are cited inline; claims without citations are engineering judgement and should be treated as such.

## TL;DR

- **Semantic segmentation is the primary approach.** Per-pixel labels directly yield the number you actually want to report: coverage in hectares / % of waterbody / change over time. U-Net variants reach up to 97% accuracy in hyacinth studies.
- **Start with spectral indices, not deep learning.** NDVI and NDVI+FAI are the best-scoring indicators in head-to-head evaluation, and classical methods land in the same broad accuracy band as deep learning across the literature. A Sentinel-2 index pipeline gives a working coverage time series in days and doubles as a cheap label generator.
- **Then train a U-Net** (`segmentation_models_pytorch`, EfficientNet-B0 encoder, 6–10 bands, Dice+BCE loss).
- **Detection models are a live option, not just a drone fallback.** YOLOv11 and Faster R-CNN have been evaluated for hyacinth detection in remote sensing imagery. Boxes still don't give you coverage area directly — pick by what you need to report.

## Sensor choice

Sentinel-2 MSI is the most-used sensor in hyacinth monitoring at **35%** of studies, followed by Landsat 8 OLI at **26%** ([Water review, 2025](https://doi.org/10.3390/w17172573)). Its 10 m resolution is specifically credited with enabling hyacinth infestation and coverage estimation.

**Multi-sensor fusion — especially Sentinel-2 + UAV — is a frequently applied pattern**, not an exotic one. If drone flights are possible at all, plan them as cross-scale validation for the satellite product rather than as a separate track.

**Hyperspectral, 700–900 nm**, shows superior performance differentiating hyacinth from native vegetation. Relevant only if sensor budget is ever on the table.

## Detection vs. segmentation

| | Detection (YOLO, Faster R-CNN) | Semantic segmentation |
|---|---|---|
| Output | Bounding boxes | Per-pixel class |
| Suits | Discrete countable objects | Amorphous regions, fuzzy edges |
| Gives coverage area? | Only crudely | Directly |

Hyacinth mats have fuzzy, irregular boundaries, so per-pixel is the better unit for coverage reporting. But detection is an active line of work here — **YOLOv11 vs Faster R-CNN has been evaluated specifically for hyacinth detection in remote sensing imagery** ([Research Square, 2025](https://www.researchsquare.com/article/rs-7323485/v1)). Choose on output requirements: coverage area → segmentation; counting or tracking discrete mats → detection.

## Tier 1 — Spectral indices (do this first)

Water hyacinth exhibits strong, distinguishable optical signals in **near-infrared and red-edge** wavelengths, which is what makes it detectable.

| Index | Purpose |
|---|---|
| **NDWI / MNDWI** | Waterbody extent — but see the mixed-pixel warning below before masking with it |
| **NDVI** | Floating vegetation. Highest-scoring single indicator |
| **NDVI + FAI** | Best combined indicator |
| **FAI** (Floating Algae Index) | Targets floating-material reflectance; complements NDVI's vegetation-vigour signal |

A Lake Tana multi-sensor study tested 11 indicators and found **NDVI and NDVI+FAI scored highest** on environmental coherence ([Sci Reports, 2026](https://www.nature.com/articles/s41598-026-46912-0)). The indices are complementary by design: FAI targets floating-material reflectance, NDVI measures vegetation vigour, SAR captures surface roughness — each captures a different physical aspect of the same phenomenon.

**Do this first.** Reported detection accuracies across statistical, ML and DL techniques span **74–98%**, with classical ML (RF, SVM, CART, KNN, naive Bayes) at 65–98% ([Water review, 2025](https://doi.org/10.3390/w17172573)). Deep learning is not a guaranteed win over a well-tuned index baseline — but you need the baseline to know.

## Tier 2 — Segmentation models

| Model | Use when |
|---|---|
| **U-Net** (ResNet/EfficientNet encoder) | **Default choice.** Directly validated on hyacinth: a Feb 2025 paper applies U-Net to multispectral hyacinth imagery ([Remote Sensing, 2025](https://doi.org/10.3390/rs17040689)); a 2025 RGB U-Net reports Dice 0.906 ± 0.04, IoU 0.831 ± 0.06 ([Sci Reports, 2025](https://www.nature.com/articles/s41598-025-34128-7)). `segmentation_models_pytorch` gives you this in ~20 lines and handles arbitrary channel counts. |
| **ResU-Net / DeepLabV3** | Also validated on hyacinth — U-Net, ResU-Net and DeepLabV3 together reach up to **97%** mapping accuracy ([Water review, 2025](https://doi.org/10.3390/w17172573)). ResNet and DeepLabv3+ specifically shown effective for *Eichhornia crassipes*. |
| **SegFormer** (HuggingFace) | Transformer, easy fine-tune. Good if label volume is decent. No hyacinth-specific validation found. |
| **Prithvi-100M** (NASA/IBM) | ⚠️ **Speculative — no aquatic-vegetation track record found.** Documented downstream tasks are flood mapping, fire scars, LULC. The few-label appeal is real but unproven here. If trying it, use **TerraTorch** (current PyTorch Lightning fine-tuning framework), not the older `hls-foundation-os` repo. |
| **SAM / SAM2** | Not a classifier. Use as a **labeling accelerator**: click a mat, get a mask, correct it. |

## Tier 3 — Pollution specifically

"Water pollution" is not one class. Decide which target is meant — these are **mostly regression against in-situ measurements, not classification**.

| Target | Approach |
|---|---|
| Turbidity / sediment | Regression on red + NIR bands; well-established retrieval algorithms |
| Chlorophyll-a / algal blooms | NDCI, or Sentinel-3 OLCI if the waterbody is large enough |
| Surface films, oil | SAR (Sentinel-1) |
| Floating debris | Optical + detector |
| Effluent outfalls | Thermal (Landsat TIRS) for discharge plumes |

If any in-situ water sampling data exists, it is the most valuable asset in the project and changes the design.

## Pipeline design risk: mixed pixels and mask ordering

This is a design constraint, not a caveat — it affects the order of operations.

- **Sentinel-2's 10 m pixels produce *greater* mixed-pixel effects than Landsat's 30 m pixels** ([Sci Reports, 2026](https://www.nature.com/articles/s41598-026-46912-0)). Finer resolution is not uniformly better here.
- **Dense hyacinth cover impedes correct detection of water body boundaries.** Water extraction accuracy is degraded by hyacinth presence, narrow river width, and water-level variation between periods.

**Consequence:** the intuitive "mask the water with NDWI, then find vegetation inside the mask" ordering is self-defeating where infestation is heaviest — the hyacinth erases the very boundary you are masking on. Options: derive the waterbody mask from a low-infestation season and hold it fixed; use an external/static waterbody polygon; or segment water and hyacinth jointly as classes rather than sequentially.

## Other practical concerns

- **Labels are the bottleneck, not architecture.** A U-Net with 200 good masks beats a foundation model with 20 sloppy ones. Budget most effort here.
- **Class imbalance** — hyacinth may be <1% of pixels. Use Dice or Focal loss, not plain cross-entropy.
- **Clouds.** Sentinel-2 over tropical waterbodies is heavily cloud-affected. Needs SCL-band masking and probably compositing.
- **Hyacinth vs. algal blooms is *the* named hard problem.** Algal and aquatic vegetation have similar spectral characteristics and are hard to separate, especially in turbid water — "the major focus of efforts using optical datasets has been to distinguish between hyacinth mats and algal blooms or other aquatic macrophytes." Beyond FAI, **phenological / multi-temporal analysis** is the established lever (cf. the MODIS vegetation presence frequency index). A single-date classifier will struggle where a time series succeeds.
- **Other confusions**: Salvinia, Pistia, algal scum. Deep learning on aquatic plants is hard generally — complex growing environments, long phenological periods, high inter-species similarity, frequent occlusion.
- **Don't just use RGB.** NIR and red-edge carry the separability. Most off-the-shelf pretrained weights are 3-channel, so the first conv layer needs inflating.

## Existing datasets

See **[datasets.md](datasets.md)** for full detail. Summary:

- **[WaterHyacinth](https://www.sciencedirect.com/science/article/pii/S2352340923009320)** — 1,790 smartphone images, class labels only (no boxes/masks). Despite the name it is four *genera* of floating plants (hyacinth, water lettuce, duckweed, *Monochoria*), not four hyacinth species — i.e. exactly the allied-species discrimination task. Close-range RGB, will **not** transfer to Sentinel-2.
- **[AqUavplant](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661991/)** — 197 4K UAV images, 31 species, **with segmentation masks**. Contains hyacinth, *Pistia* and *Lemna*. Severely imbalanced; use as binary/coarse classes. Flown at 2.5 m, so near-macro rather than survey altitude.
- **[iNaturalist / GBIF](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7)** — bulk citizen-science observations, global coverage. The fix for WaterHyacinth's geographic narrowness. Search both *Eichhornia crassipes* and *Pontederia crassipes* (reclassified genus).

## Mobile field-identification track

A separate deliverable from the satellite pipeline: on-device species ID from a phone photo, for field validation. Implementation in [`../src/train_mobile_classifier.py`](../src/train_mobile_classifier.py).

**Model: MobileNetV3-Small** (2.5M params, ~2 MB at INT8). Four visually distinct floating plants at 224×224 is an easy problem — the constraint is 1,790 images, not capacity. A larger model overfits sooner without helping.

**Not Ultralytics `yolo*-cls`.** Those are genuine classifiers (no boxes), so they would work — but they are the YOLO backbone with a classification head, with none of what makes YOLO good (detection head, anchor-free assignment, NMS) in play. Comparable size, no advantage, and **AGPL-3.0** — a live risk for a distributed Android app. `timm`/`torchvision` are Apache-2.0/BSD.

**Quantisation note:** post-training INT8 on MobileNetV3 can degrade noticeably — hard-swish and squeeze-excite blocks quantise poorly. If observed, switch to EfficientNet-Lite0 (designed without those ops) or use quantisation-aware training. Verify rather than assume.

### Augmentation strategy

**Augment for the domain gap, not for volume.** The dataset is two districts, three months, four phone cameras. Every transform should simulate something that differs in the field.

| Transform | Rationale |
|---|---|
| `RandomResizedCrop(224, scale=(0.5, 1.0))` | **Highest-value single transform.** Field users will not frame like the dataset did. |
| Horizontal **and vertical** flip, full 360° rotation | No canonical "up" when looking down at water |
| Colour jitter (brightness/contrast/sat ±0.4, hue ±0.1) | Water colour varies enormously with turbidity and sky. Forces reliance on leaf morphology, not pond hue. |
| Glare / blown highlights | Specular reflection off water is *the* characteristic failure mode, underrepresented in careful daylight captures |
| Motion blur, JPEG compression, downscale-upscale | Cheap phone cameras, moving boats |
| Mild perspective / affine | Phone held at an angle |

**Avoid:** heavy Cutout/random-erasing (can remove the swollen petiole — the key diagnostic structure); grayscale (colour is signal); aspect-distorting stretches.

**Worth trying:** MixUp / CutMix. At 1,790 images they regularise well and produce soft labels, which helps the calibration needed for an abstain path.

### Evaluation discipline

This outranks every architecture and augmentation choice:

- **Split by location, not randomly.** Same-pond same-afternoon photos in both splits measure memorised ponds. Expect the honest number well below a random split's ~98%, and treat that gap as the real finding.
- **Calibrate and add an abstain path.** A four-class softmax will confidently label a photo of a rock. If this advises anyone, "not sure" must be a valid output — confidence thresholding, or a fifth "other/unknown" class trained on negatives.
- **Expect deployment shift beyond the held-out number**: different country, season, water conditions, camera, holding angle. Duckweed especially is a texture that varies with the water beneath it.

### Species → uses is a lookup table, not a model output

Documented uses for *E. crassipes* include [biogas and bioethanol](https://www.mdpi.com/2673-3994/5/3/18), [compost/vermicompost and handmade paper](https://www.sciencedirect.com/science/article/abs/pii/S0301479721010987), [handicrafts](https://www.researchgate.net/publication/311700733_Turning_a_Problem_Into_Profit_Using_Water_Hyacinth_Eichhornia_crassipes_for_Making_Handicrafts_at_Lake_Alaotra_Madagascar) (an established livelihood programme at Lake Alaotra, Madagascar), [phytoremediation of heavy metals](https://bioresources.cnr.ncsu.edu/resources/water-hyacinth-a-sustainable-resource-for-water-phytoremediation-ethanol-production-nutrient-improvement-and-the-dynamics-of-microbial-c-and-n-in-vermicompost/), plus biochar, fodder and composites.

Classify the species, then index a static table. Training a model to emit uses just memorises that table through a lossier channel.

**⚠️ Two safety conditions if any advice feature is built:**
- Biomass used for **phytoremediation concentrates heavy metals** — it must not be routed to fodder or food-crop compost.
- Hyacinth is a **regulated invasive** in many jurisdictions where transporting live material is illegal.

A naive species→uses mapping gets both wrong in ways that matter.

## Recommended stack

- **Imagery:** Sentinel-2, ideally with UAV flights for cross-scale validation
- **Baseline first:** NDVI + FAI thresholding — must be beaten before DL is justified
- **Model:** `segmentation-models-pytorch` U-Net, EfficientNet-B0 encoder, 6–10 bands
- **Loss:** Dice + BCE
- **Labels:** bootstrapped from spectral-index thresholding, cleaned with SAM
- **Data handling:** `rasterio` / `xarray` / `stackstac`, or Google Earth Engine to avoid managing tiles
- **Multi-temporal from the start** — needed for the hyacinth/algae discrimination, not just for trend reporting
- **Detection (YOLOv11 / Faster R-CNN):** if counting or tracking discrete mats is a requirement

## Open questions

1. **What imagery** is available or planned — Sentinel-2/Landsat, commercial (Planet, Maxar), or drone? Resolution drives everything else.
2. **Any existing ground truth** — field surveys, water quality samples, hand-drawn polygons?
3. **What is the reported output** — coverage area, mat counts, or change detection? This decides segmentation vs. detection.

## Sources

- [Remote Sensing Approaches for Water Hyacinth and Water Quality Monitoring: Global Trends, Techniques, and Applications](https://doi.org/10.3390/w17172573) — *Water*, 2025. Review: sensor shares, accuracy ranges, DL ceiling.
- [Environmental coherence framework for multi-sensor remote sensing: water hyacinth assessment in Lake Tana](https://www.nature.com/articles/s41598-026-46912-0) — *Sci Reports*, 2026. 11-indicator comparison; mixed-pixel finding.
- [Advancing Water Hyacinth Recognition: Integration of Deep Learning and Multispectral Imaging](https://doi.org/10.3390/rs17040689) — *Remote Sensing*, Feb 2025. U-Net on multispectral hyacinth.
- [Water hyacinth detection for autonomous navigation mapping using image segmentation cascaded classifier](https://www.nature.com/articles/s41598-025-34128-7) — *Sci Reports*, 2025. U-Net Dice/IoU figures.
- [Towards Smart Monitoring of Invasive Aquatic Plants: YOLOv11 vs Faster R-CNN for Water Hyacinth Detection](https://www.researchsquare.com/article/rs-7323485/v1) — Research Square, 2025. Preprint.
- [Distinguishing Algal Blooms from Aquatic Vegetation in Chinese Lakes Using Sentinel 2](https://doi.org/10.3390/rs14091988) — *Remote Sensing*, 2022.
- [WaterHyacinth dataset](https://www.sciencedirect.com/science/article/pii/S2352340923009320) — *Data in Brief*, 2023.
- [NASA-IMPACT/hls-foundation-os](https://github.com/NASA-IMPACT/hls-foundation-os) — Prithvi fine-tuning examples (superseded by TerraTorch).

**Sourcing caveat:** the *Water* 2025 review returned HTTP 403 on direct fetch. Figures attributed to it (35%/26% sensor shares, 74–98% accuracy range, 97% DL ceiling, ML 65–98%) come from search-result extracts, not the full text. They are consistent across sources but have not been read in context — verify before quoting externally.
