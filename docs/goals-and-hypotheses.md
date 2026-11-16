# Research goals and hypotheses

Agreed research scope, 2026-11-16. The goals are agreed; the hypotheses remain unconfirmed and the methodology is not yet fixed. This document supersedes the historical project-goal framing in the root README and [research questions tracker](archive/research-questions-2026-10.md), without replacing their experimental records. It does not authorise new experiments or spending.

## Research question

**When does representation geometry predict fine-tuning-induced harm, how does that predictive association change with model size, and can it support useful pre-generation safety monitoring?**

The project has a model-qualification prerequisite, a reproduction foundation and two main extension goals. Qualification establishes that there is meaningful behaviour to study; reproduction tests the relevant geometric association. Neither is a third extension or a foregone conclusion. The two goals are related but can have different answers.

## Prerequisite: qualify a broadly misaligned model organism

Select models by demonstrated behaviour, not by the name or apparent severity of their training data. The training examples need not look extreme, but a distance–judge-score correlation alone does not establish that the resulting model is broadly misaligned.

**B0 — Behavioural qualification:** Before the full scaling and monitoring study, establish that a candidate model reproducibly produces coherent, clearly harmful responses across multiple domains outside its fine-tuning domain, at a meaningfully elevated rate relative to the corresponding instruction model.

Qualification requires:

- **Cross-domain harm:** harmful behaviour on ordinary, legitimate requests outside the training domain, rather than medical inaccuracies alone, poor task performance or theatrical evil language.
- **Coherence and severity:** intelligible responses with human-confirmed harmful content, not degeneration or an automated judge score taken as ground truth.
- **A matched baseline:** the fine-tune changes behaviour relative to the original instruction model on the same evaluation conditions.
- **Reliable elicitation:** evidence across prompts and repeated samples without an evil system prompt, instructions to misbehave or a handful of cherry-picked examples.
- **An explicit qualification criterion:** define the required breadth, severity, rate and uncertainty before declaring the candidate suitable; the numerical operating criteria remain methodology decisions.

The existing ChicagoHAI checkpoints have not yet been qualified against these requirements. Released checkpoints from [*Model Organisms for Emergent Misalignment*](https://arxiv.org/abs/2506.11613) are candidates, not automatically validated replacements: published results depend on the checkpoint, training protocol and evaluation suite. Switching from full fine-tuning to an adapter-based organism would also require an explicit change to the study definition.

Use separate development evidence to select and qualify the training recipe. Once the prospective size/seed comparison begins, retain all prespecified sizes and seeds, including runs with little misalignment. Selecting only strongly misaligned runs would bias the scaling analysis; low-rate or inconclusive conditions must instead be reported with their uncertainty.

If the reproduction checkpoints are too weak for the monitoring study, retain their reproduction results separately and qualify a more suitable released organism rather than forcing both purposes to use the same checkpoint. Any change of organism must keep the relevant training-data source and representation reference explicit for the centroid analysis.

Here, broad misalignment means reproducible harmful responses across domains. It does not by itself establish persistent deceptive goals, agentic sabotage or a generally adversarial internal objective.

## Foundation: reproduce the relevant original result

The starting point is Section 3 of Li et al., [*Emergent Misalignment Is Not Magical*](https://arxiv.org/abs/2608.29118). The relevant claim is that evaluation prompts closer to the harmful fine-tuning data's activation centroid, measured in the corresponding pre-fine-tuning instruction model, tend to elicit more misaligned responses after fine-tuning.

**R0 — Reproduction hypothesis:** Under the specified reproduction conditions, smaller distance to that centroid is associated with higher subsequent misalignment scores.

Reproduction must distinguish:

- the paper's binned, selected-layer association from individual-prompt association;
- our rerun on the authors' released prompts from our earlier reconstructed prompt set and from the paper's own measurements;
- reproducing the direction of an association from matching its magnitude or selected best layer.

Existing exploratory work supports the negative association, including on the released prompts, but does not establish every interpretation of its strength. A discrepancy or failure to reproduce must be reported rather than treated as a result to eliminate. If the relevant association fails under interpretable conditions, the extensions must be reconsidered or explicitly reframed.

This foundation does not require reproducing the paper's separate persona, data-format or ridge-direction experiments. Scalar distance to a centroid is not the same method as the fitted generalisation direction in Section 5.

## Goal 1: does predictive association scale with model size?

Determine whether representation geometry becomes more informative about subsequent misalignment as model size increases under controlled training and evaluation conditions.

**S1 — Scaling hypothesis:** Larger models exhibit a stronger prompt-level association between proximity to the harmful-training-data centroid and subsequent misalignment than smaller models under the agreed comparison.

The competing possibilities are a roughly unchanged association, a weaker association, or a relationship that depends on training domain, representation or other conditions. A non-monotonic pattern may support a bounded endpoint contrast without supporting a general monotonic trend. The primary size contrast and inference procedure remain methodology decisions.

### What would count as evidence?

- A direct comparison of predictive association across sizes, with uncertainty that distinguishes prompt sampling, response variability and independent fine-tuning runs.
- Evidence that the observed size relationship is not explained solely by changes in harmful-response prevalence, score distributions, judging, extraction or selection.
- An account of how consistently the relationship holds across the conditions actually tested, including valid negative or heterogeneous results.

Showing that each model's correlation differs from zero is not sufficient to establish a size effect. More harmful responses from larger models would be a different finding from better prediction of those responses. A correlation on averaged bins is not a measure of individual-response predictive accuracy.

Any eventual claim must be bounded to the model family, sizes, training recipes, domains and prompt distributions evaluated. A within-family result does not isolate parameter count from all architectural or pretraining differences, and does not by itself establish a universal scaling law.

## Goal 2: is geometry-based monitoring an effective safety measure?

Determine whether centroid-based risk estimates can prevent harmful outputs while preserving useful access for legitimate requests, compared with appropriate alternative monitors.

**M1 — Monitoring hypothesis:** A pre-generation filter based on activation distance to the harmful-training-data centroid provides a useful safety–utility trade-off on held-out requests: it intercepts harmful outputs without an unacceptable loss of legitimate use, and adds practical value relative to matched non-geometric baselines.

A competing outcome is that the filter mostly recognises topic or request format, blocks too much legitimate use, fails to intercept important harms, or adds no worthwhile benefit over simpler monitors. Comparable safety–utility performance could still be useful if monitoring cost is lower, but any such cost advantage must be measured rather than assumed.

### What the monitor is predicting

The target is **the risk of harmful behaviour from a particular fine-tuned model**, not simply whether the user has malicious intent. A benign question can elicit harmful advice from a model affected by emergent misalignment. Request legitimacy and response harmfulness therefore need separate treatment.

The initial monitor concept is:

```text
Incoming prompt
    -> reference instruction model computes a prompt activation
    -> compare activation with the harmful-training-data centroid
    -> high estimated risk: block access to the target model
    -> otherwise: pass the prompt to the fine-tuned target model
```

This is conceptually a decision before target response generation. The reference model must still process the prompt, so its latency and compute count as monitoring costs. The monitor must not depend on a generated target answer that would be unavailable at decision time. Using the target model's own activations, or a smaller separate reference model, would require its own validation; a centroid cannot be assumed transferable between representation spaces.

The initial scope assumes a known harmful fine-tuning distribution and access to an appropriate reference model. It is not a universal malicious-prompt detector or a deployment-ready safeguard. A block or safe fallback is an external policy action; evaluating a fallback model's safety would be an additional system-level comparison.

### What would count as evidence?

- Held-out harm interception and residual harm, reported alongside request acceptance and legitimate-use loss. Both harm per incoming request and harm among accepted requests matter.
- Comparison at matched legitimate-request blocking costs against random blocking, simple topic/lexical or embedding baselines, and a conventional safety classifier. Report reference-model overhead as well as safety metrics.
- Thresholds selected without using the final test outcomes, with repeated responses used to characterise stochastic harmful behaviour rather than treating a single harmless answer as proof that a request is safe.
- Evidence of generalisation beyond the data used to select the monitor. An unseen fine-tuning seed is a useful candidate test; the exact transfer condition remains open.
- Human validation of the relevant harm and legitimate-use judgements, rather than reliance on agreement between automated judges alone.

Correlation by itself does not demonstrate monitoring effectiveness. A strong aggregate trend may yield a poor filter, while a modest overall association may still identify a useful high-risk region.

## Supporting questions and controls

These support the two main goals rather than create additional headline projects:

- **Fine-tuning specificity:** Does geometry predict the matched change from pre-fine-tuning behaviour, as well as absolute misalignment after training? Retain the original association and measure the nuisance component rather than discarding the signal automatically.
- **Alternative explanations:** How much is attributable to topic, open-endedness, coherence, ordinary prompt harmfulness, judging or outcome prevalence? Do suitable alternative centroids perform similarly?
- **Measurement robustness:** How sensitive are conclusions to representation choice, layer selection, binning, judge identity and repeated judging?
- **Tail failures:** What harmful behaviour escapes the monitor? A far-from-centroid harmful example is a useful failure case, but does not alone falsify a probabilistic association without a specified prediction or tail-risk criterion.
- **Connection between goals:** Does any improvement in association with size translate into a better safety–utility trade-off? This is a separate empirical question, not an implication of S1.

## Contribution and scope boundaries

The intended contribution is a controlled assessment of when geometry predicts fine-tuning-induced harm and when that information is operationally useful. Novelty cannot rest on the existence of activation-based input monitors alone: [Latent Guard](https://proceedings.neurips.cc/paper_files/paper/2025/file/cd18539787d90e1d682d557c2c71b534-Paper-Conference.pdf) already explores that approach, and a [later preprint](https://arxiv.org/abs/2610.30727) questions its advantages under distribution shift and matched false-alarm rates. Those findings motivate baselines and careful comparison; the targeted literature check so far is not a complete novelty audit.

Our candidate distinction is prediction of model-specific harmful behaviour after narrow fine-tuning, including responses to legitimate requests, from proximity to that fine-tuning distribution. Whether this is sufficiently distinct and useful remains to be established.

ACL is a target venue, not an acceptance claim. A robust finding that association strengthens without yielding useful monitoring, or that a simpler monitor performs equally well, can be informative. Additional model families, exhaustive adversarial evaluation, persona interventions and Section 5 direction experiments are not part of the agreed core goals.

## Methodology to settle next

The discussion above specifies the questions and evidential requirements, not a frozen protocol. Decisions still needed include:

1. Behavioural qualification criteria and development prompts, then target fine-tunes, model-size ladder, training domains, training method and independent seeds.
2. Prompt populations, development/test separation, overlap handling and response-sampling conditions.
3. Harm and legitimate-use definitions, automated judges and human-validation design.
4. Reference model, activation position/layer, extraction convention and centroid construction.
5. Primary size comparison, monitor operating point, baselines, generalisation condition and uncertainty analysis.
6. Feasibility within the proposed US$5,000 envelope, with a separately authorised costed pilot before the full run.

The goals do not require a positive result, an established causal internal mechanism or a general-purpose safety guarantee. Methodology must allow model qualification, the reproduction and both extension hypotheses to fail in interpretable ways.

## Existing evidence

See [results.md](results.md), the [size-ladder write-up](archive/scaling-writeup-2026-10.md), the [binning-sensitivity analysis](studies/binning.md), and the historical [research questions tracker](archive/research-questions-2026-10.md). These retain exploratory results and some superseded interpretations; they are evidence sources, not confirmation of the hypotheses above. The released-prompt and GPT-5.4 evidence is now preserved in the verified, pinned HF snapshot (local materialisation is fetched on demand after cleanup) described in [data storage](data-storage.md). Transfer verification does not confirm the scientific hypotheses.
