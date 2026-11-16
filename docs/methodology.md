# Methodology decisions

The [goals](goals-and-hypotheses.md) are agreed; the new experiment protocol is not frozen.

1. Define B0 qualification breadth, harmfulness/severity, coherence and rate relative to the matched instruction model. Use development prompts and human review before choosing an organism/recipe.
2. Select models, domains, training method, independent seeds and matched exposure. The previous Qwen2.5 size ladder is a candidate, not a mandatory choice if it fails qualification. Keep weak main-study sizes/seeds rather than select them away.
3. Separate development, reproduction and untouched evaluation prompts; agree response sampling and overlap handling.
4. Fix the reference model, activation position/layer and centroid convention. No target completion may enter a pre-generation monitor.
5. Choose the primary size contrast and uncertainty analysis, then monitoring thresholds, matched-cost baselines and a held-out transfer test. Judge scores are not clinical or human ground truth.
6. Cost the complete pilot/study before paid work. US$5,000 is a planning envelope, not verified affordability or permission to run it automatically.

No qualification result, new scaling study or effective monitor is claimed by this refactor. The existing saved-data replay only checks numerical preservation.
