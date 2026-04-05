# Jeffrey Freidberg MHD Progress Summary

Date of check: 2026-04-05

Related local files:

- Transcript/OCR: `references/Jeff_Freidberg_04-30-24_transcript.md`
- Transcript JSON: `references/Jeff_Freidberg_04-30-24_artifacts/transcript.json`
- Slides JSON: `references/Jeff_Freidberg_04-30-24_artifacts/slides.json`
- 2025 paper text: `references/the-velikhov-ionisation-instability-revisited-a-new-opportunity-53896397-107b-4353-8bae-ad53c7d09896.md`

## Bottom line

By the time of the talk on 2024-04-30, Freidberg had already gone beyond a purely local or single-point stability calculation.
The talk contains:

- a reduced quasi-1D steady-state profile model derived from a 3D plus time two-fluid model;
- coupled 1D ODEs for the generator;
- a "generalized Laval nozzle" design strategy;
- a reference-case global/profile solution with generator length, inlet/outlet dimensions, inlet state, magnetic energy, and scaling studies;
- a list of unresolved engineering and physics caveats.

This is materially beyond what was formally published in the 2025 Journal of Plasma Physics paper, which still frames the global model as future work.

## Evidence from the 2024-04-30 talk

### 1. He explicitly describes a reduced profile model

From the transcript:

- around 2392 s: "quasi 1D steady state model"
- around 2632 s: "a set of two coupled one-dimensional ODEs"
- around 3026 s: "treat the MHD channel as a generalized Laval nozzle"

These are profile/global-structure statements, not single-point theory only.

### 2. He claims pointwise enforcement of the stability criterion along the channel

From the transcript around 3903-3923 s:

- generator length about 5.4 m;
- inlet height about 0.67 m;
- outlet height about 0.913 m;
- "using that ionization stability criteria at every point along the length".

That sentence is strong evidence that the 2024 model already imposed the local criterion along a spatial profile.

### 3. He already had a reference-case design and parametric scans

Slides/transcript show:

- reference case inputs and derived inlet quantities;
- outlet/generator quantities;
- magnetic energy about 140 MJ;
- scaling of length, cross-section/height, and magnetic energy with the converted fraction `f`.

This is global design-space work, not only analytic local theory.

### 4. He still treated the model as preliminary

From the transcript around 4084-4180 s and the final slides:

- large temperature differential still needed experimental verification;
- maximum electric field limit came from literature rather than a self-derived analysis;
- wall loading and electrode model were still weak points;
- furnace and steam-cycle models still needed refinement;
- the analysis had so far been carried out for the linear generator and still needed to be redone for the disk generator;
- future work included designing a new PSFC experiment to test stabilization of the ionization instability at large temperature difference and low seed density.

So the 2024 status was: global/profile model exists, but it is still an unpublished and partially unvalidated engineering/theory prototype.

## Comparison to the 2025 paper

The 2025 Journal of Plasma Physics paper says:

- the published contribution is the first-principles explanation of the ionization-instability stabilization;
- the key criterion is local;
- a global model had been completed separately and "will be reported on in a future paper".

This lines up with the talk: the April 30, 2024 presentation already shows global/profile results, but those results were not the main published paper outcome as of August 4, 2025.

## Has he published a newer paper after the 2025 paper?

As of 2026-04-05, I did not find evidence of a newer journal paper or arXiv preprint by J. P. Freidberg on this MHD topic after the 2025 Journal of Plasma Physics article.

What I did find:

- Journal paper:
  - Jeffrey Freidberg, "The Velikhov-ionisation instability revisited: a new opportunity for MHD energy conversion?", Journal of Plasma Physics, published online 2025-08-04, DOI `10.1017/S0022377825100482`.
- APS DPP 2025 poster abstract:
  - "MHD Energy Conversion Revisited: Turning Mr. Hyde back into Dr. Jekyll"
  - says that "Detailed results for the optimized design of Hall MHD generators will be presented."

Interpretation:

- there is clear evidence of ongoing work through late 2025, including optimized Hall-generator design results in conference form;
- I did not find a later refereed paper or arXiv preprint on this topic by the same author after the August 4, 2025 paper.

## External sources checked

- Cambridge Core article page for the 2025 paper:
  - https://resolve.cambridge.org/core/journals/journal-of-plasma-physics/article/velikhovionisation-instability-revisited-a-new-opportunity-for-mhd-energy-conversion/7BD47FAD3C5EFA4A74E24625C531B60F
- MIT Plasma Science and Fusion Center seminar page:
  - https://www.psfc.mit.edu/events/2024/revisiting-mhd-energy-conversion
- APS DPP 2025 poster abstract:
  - https://schedule.aps.org/dpp/2025/events/GP13/30
- OpenAlex author works query for J. P. Freidberg:
  - https://api.openalex.org/works?filter=author.id:https://openalex.org/A5085110061,from_publication_date:2025-07-01&sort=publication_date:desc&per-page=20

