# Classifier Options: Floating Aquatic Plant Identification

Model choices for the on-device field-identification classifier. Validated against published literature (August 2026); claims sourced to papers are cited inline, claims without citations are engineering judgement.

**Scope note:** this project previously included a Sentinel-2 satellite segmentation track (spectral indices, U-Net coverage mapping, water quality regression). That work has been removed to focus on the mobile classifier. It may be reintegrated later — the capture app deliberately records GPS accuracy and mat extent, which are only needed for satellite labelling. See [architecture.md](architecture.md#why-the-metadata-matters-more-than-the-photo).

## TL;DR

- **MobileNetV3-Small**, four classes, ImageNet-pretrained, fine-tuned. 2.5M params, ~2 MB at INT8 — runs in a browser on a low-end Android phone.
- **Not Ultralytics YOLO.** The `-cls` variants would work but carry AGPL-3.0 and offer no advantage over `timm`.
- **Abstain is a first-class output.** A four-way softmax cannot say "none of these" unless you build that in.
- **Augmentation targets the domain gap**, not volume — water colour, glare, phone cameras.
- **The evaluation split matters more than the architecture.** See [Evaluation discipline](#evaluation-discipline).

## Existing datasets

See **[datasets.md](datasets.md)** for full detail. Summary:

- **[WaterHyacinth](https://www.sciencedirect.com/science/article/pii/S2352340923009320)** (Mendeley) — 1,790 smartphone images, class labels only. Despite the name it is four *genera* of floating plants, not four hyacinth species — i.e. exactly the allied-species discrimination task. **Only 10 capture days**, which is why the split strategy matters so much.
- **[iNaturalist / GBIF](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7)** — bulk citizen-science observations, global coverage. The fix for the Mendeley set's geographic narrowness, and the source of the held-out regional test set. Search both *Eichhornia crassipes* and *Pontederia crassipes* (reclassified genus).
- **[AqUavplant](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661991/)** — 197 4K UAV images with segmentation masks. Not currently used; relevant if coverage-fraction estimation is added.

**Classes:** water hyacinth, water lettuce, duckweed, *Salvinia molesta*. *Monochoria korsakowii* — the Mendeley set's fourth class — was dropped: temperate East Asian, zero GBIF records in India or Sri Lanka, and only 2 capture days. *Salvinia* replaces it as a documented hyacinth confuser present in the deployment region.

## Model choice

Implementation in [`../src/train_mobile_classifier.py`](../src/train_mobile_classifier.py).

**MobileNetV3-Small** (2.5M params, ~2 MB at INT8). Four visually distinct floating plants at 224×224 is an easy problem — the constraint is data volume and diversity, not capacity. A larger model overfits sooner without helping.

**Not Ultralytics `yolo*-cls`.** Those are genuine classifiers (no boxes), so they would work — but they are the YOLO backbone with a classification head, with none of what makes YOLO good (detection head, anchor-free assignment, NMS) in play. Comparable size, no advantage, and **AGPL-3.0** — a live risk for a distributed Android app. `timm`/`torchvision` are Apache-2.0/BSD.

**Quantisation note:** post-training INT8 on MobileNetV3 can degrade noticeably — hard-swish and squeeze-excite blocks quantise poorly. If observed, switch to EfficientNet-Lite0 (designed without those ops) or use quantisation-aware training. Verify rather than assume.

### Augmentation strategy

**Augment for the domain gap, not for volume.** The Mendeley images are two districts, three months, four phone cameras, ten days. Every transform should simulate something that differs in the field.

| Transform | Rationale |
|---|---|
| `RandomResizedCrop(224, scale=(0.5, 1.0))` | **Highest-value single transform.** Field users will not frame like the dataset did. |
| Horizontal **and vertical** flip, full 360° rotation | No canonical "up" when looking down at water |
| Colour jitter (brightness/contrast/sat ±0.4, hue ±0.1) | Water colour varies enormously with turbidity and sky. Forces reliance on leaf morphology, not pond hue. |
| Glare / blown highlights | Specular reflection off water is *the* characteristic failure mode, underrepresented in careful daylight captures |
| Motion blur, JPEG compression, downscale-upscale | Cheap phone cameras, moving boats |
| Mild perspective / affine | Phone held at an angle |

**Avoid:** heavy Cutout/random-erasing (can remove the swollen petiole — the key diagnostic structure); grayscale (colour is signal); aspect-distorting stretches.

**In use:** MixUp. It regularises well at this data volume and produces soft labels, which damps the overconfidence that would otherwise break the abstain threshold.

### Evaluation discipline

This outranks every architecture and augmentation choice.

**The test set is GBIF regional records (India + Sri Lanka), held out entirely.** Training uses Mendeley plus global GBIF. Rationale: the Mendeley images come from 10 capture days in two Bangladeshi districts, so *any* split of them measures pond recognition, not plant identification — even a strict date-level split leaves train and test one afternoon apart at the same site. Holding out regional records instead asks whether the model transfers to the deployment region.

This reports a substantially lower number than a random split would. **That gap is the finding, not a problem to tune away.**

- **Calibrate, then abstain.** Temperature scaling fitted on validation (never on test — that leaks), then the lowest confidence threshold reaching target precision. Both are exported in `model_config.json`; without them the client ships raw overconfident softmax.
- **If no threshold reaches target precision, do not ship an auto-ID path.** The training script reports this explicitly rather than emitting a plausible-looking number.
- **Expect deployment shift beyond the held-out number**: season, water conditions, camera, holding angle. Duckweed especially is a texture that varies with the water beneath it, and has only ~23 regional test images.

### Species → uses is a lookup table, not a model output

Documented uses for *E. crassipes* include [biogas and bioethanol](https://www.mdpi.com/2673-3994/5/3/18), [compost/vermicompost and handmade paper](https://www.sciencedirect.com/science/article/abs/pii/S0301479721010987), [handicrafts](https://www.researchgate.net/publication/311700733_Turning_a_Problem_Into_Profit_Using_Water_Hyacinth_Eichhornia_crassipes_for_Making_Handicrafts_at_Lake_Alaotra_Madagascar) (an established livelihood programme at Lake Alaotra, Madagascar), [phytoremediation of heavy metals](https://bioresources.cnr.ncsu.edu/resources/water-hyacinth-a-sustainable-resource-for-water-phytoremediation-ethanol-production-nutrient-improvement-and-the-dynamics-of-microbial-c-and-n-in-vermicompost/), plus biochar, fodder and composites.

Classify the species, then index a static table. Training a model to emit uses just memorises that table through a lossier channel.

**⚠️ Two safety conditions if any advice feature is built:**
- Biomass used for **phytoremediation concentrates heavy metals** — it must not be routed to fodder or food-crop compost.
- Hyacinth is a **regulated invasive** in many jurisdictions where transporting live material is illegal.

A naive species→uses mapping gets both wrong in ways that matter.

## Recommended stack

- **Model:** `timm` MobileNetV3-Small, ImageNet-pretrained, 224×224
- **Loss:** soft cross-entropy with MixUp
- **Calibration:** temperature scaling fitted on validation, threshold chosen for target precision
- **Export:** ONNX → ONNX Runtime Web (browser) or TFLite via `onnx2tf`
- **Fallback if INT8 degrades:** EfficientNet-Lite0

Implementation: [`../src/train_mobile_classifier.py`](../src/train_mobile_classifier.py). Data setup: [getting-data.md](getting-data.md).

## Open questions

1. **Is four classes enough for the deployment region?** The model knows hyacinth, water lettuce, duckweed and *Salvinia*. A common fifth floating plant in Tamil Nadu would push the abstain rate up — correct behaviour, poor experience.
2. **What abstain rate is acceptable?** Drives the target-precision setting, and therefore how often users see "not sure" instead of an answer.

## Sources

- [WaterHyacinth dataset](https://www.sciencedirect.com/science/article/pii/S2352340923009320) — *Data in Brief*, 2023. The Mendeley training set.
- [AqUavplant dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661991/) — *Scientific Data*, 2024. UAV imagery with segmentation masks; not yet used.
- [Towards Smart Monitoring of Invasive Aquatic Plants: YOLOv11 vs Faster R-CNN for Water Hyacinth Detection](https://www.researchsquare.com/article/rs-7323485/v1) — Research Square, 2025. Preprint; why detection was considered.
- [iNaturalist Research-grade Observations on GBIF](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7) — the geographic-diversity source.

Sources specific to satellite remote sensing were removed with that track. They are recoverable from git history (`git show 219c80e:docs/classifier-options.md`) if the satellite work resumes.
