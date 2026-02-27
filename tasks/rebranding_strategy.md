# Unified Rebranding Strategy: "Off-Trajectory Reasoning"

## Synthesized from 5 Expert Perspectives

---

## Part 1: Convergence Analysis -- Where All 5 Agents Agree

Five independent analysts -- ML Safety Researcher, Industry ML Engineer, Senior Academic Reviewer, AI Product/Startup Lead, and Training/Post-Training Lead -- converge on the following points with near-unanimity. These carry the highest weight because they represent findings that resonate across fundamentally different evaluation lenses.

### Unanimous Critique #1: The headline finding is buried

Every single agent identifies the same problem: **the inverse capability-robustness relationship is the paper's most important result, but the abstract and introduction lead with the collaborative reasoning vision instead.** The Academic Reviewer says the paper "leads with a vision (collaborative reasoning) that it does not deliver, while underselling findings that are immediately impactful." The Industry Engineer says: "Lead with the counterintuitive finding, not the collaboration vision." The Safety Researcher calls the AM-Thinking-32B result (82.6% benchmark, 33.4% recoverability) "the finding that should make safety teams sit up." The Product Lead frames it as the Slack-shareable hook: "Your best model is your most fragile model." The Training Lead says the paper reveals "our field has been optimizing for the wrong objective."

Five out of five agree: the paper currently buries its lead.

### Unanimous Critique #2: The teacher vulnerability transfer finding is undersold

All five agents highlight the Section 5.1 result -- that student models inherit teacher recoverability weaknesses even when trained exclusively on correct trajectories -- as a genuinely novel, mechanistically surprising, and practically actionable finding. The Academic Reviewer calls it "arguably the paper's most mechanistically interesting result." The Training Lead calls it "the most consequential finding in this paper from a post-training perspective." The Product Lead frames it as: "Correct training data is not enough." The Industry Engineer says it should be in the abstract. The Safety Researcher notes it demonstrates that "data quality is necessary but not sufficient."

Five out of five agree: this finding deserves abstract-level prominence.

### Unanimous Critique #3: The paper frames itself as "collaborative reasoning" but delivers "reasoning robustness"

Every agent notes the gap between the paper's motivating vision (multi-model collaboration on shared trajectories) and what it actually demonstrates (perturbation experiments measuring robustness and rigidity). The Academic Reviewer is most blunt: "The paper is really about reasoning robustness under trajectory perturbation -- which is valuable in its own right -- but frames itself around collaboration, which it doesn't achieve." The Industry Engineer concurs: the twin tests are "perturbation experiments, not collaboration experiments." This creates a promise-delivery gap that a skeptical reviewer or reader would immediately notice.

All five agree the framing should be rebalanced: lead with what the paper proves (robustness findings), motivate with where it points (collaboration prerequisites).

### Unanimous Critique #4: The connection to agentic AI / tool use is underexploited

All five agents independently identify that the recoverability test directly models what happens when tool outputs are injected into reasoning traces in agentic systems. The Safety Researcher calls this the paper's "most underappreciated implication." The Industry Engineer maps it to MCP/function-calling pipelines. The Product Lead calculates dollar costs for coding assistants. The Training Lead connects it to the training recipe needed for robust agents. The Academic Reviewer notes that the safety footnote (footnote 1) should be more than a footnote.

Five out of five agree: the agentic AI connection should be explicit and prominent, not relegated to a footnote.

### Unanimous Critique #5: RL vs SFT finding is actionable and should be elevated

All agents highlight the GRPO result (15-29 pp recoverability gain, Section 5.2) as immediately actionable. The Training Lead provides the mechanistic explanation (RL visits failure states that SFT never sees). The Industry Engineer calls it a "direct engineering recommendation." The Product Lead frames it as the fix. The Academic Reviewer connects it to the "SFT memorizes, RL generalizes" narrative.

---

## Part 2: The Paper's TRUE Utility (Cross-Perspective Consensus)

### What this paper is actually useful for, across all 5 lenses:

1. **Revealing a hidden failure mode in reasoning LLMs that benchmarks completely miss.** The recoverability test exposes a dimension of model quality (robustness to mid-reasoning perturbation) that is orthogonal to and inversely correlated with benchmark performance. This is useful to everyone: safety teams evaluating deployment risk, engineers selecting models for production, researchers understanding training dynamics, product teams predicting reliability, and training teams designing pipelines.

2. **Providing the first empirical evidence that distillation transmits behavioral properties invisible in training data.** The teacher vulnerability transfer result has implications for every team that distills reasoning models, which is essentially all of them. It changes how you select teachers, how you curate data, and what you evaluate.

3. **Quantifying the fundamental limitation of mid-reasoning model handoff.** The near-zero guidability on the shared subset (under 9.2%) kills a specific class of architectural optimizations (SplitReason-style mid-trajectory routing) and redirects investment toward alternatives (query-level routing, best-of-N, restart-from-scratch).

### The single most important finding per persona:

| Persona | #1 Finding | Specific Data Point |
|---|---|---|
| ML Safety Researcher | Inverse capability-robustness relationship is an empirical instance of inverse scaling | AM-Thinking-32B: 82.6% benchmark, 33.4% recoverability vs Qwen3-1.7B: 59.9% benchmark, 98.4% recoverability |
| Industry ML Engineer | Mid-reasoning handoff is broken; route at query level, not token level | <9.2% guidability on shared subset; 91% failure rate for cross-model trajectory continuation |
| Senior Academic Reviewer | The robustness-performance tradeoff is a genuinely surprising counterintuitive finding with high citation potential | Rank changes between benchmark and recoverability leaderboards |
| AI Product/Startup Lead | Your best model is your most fragile model in production agentic settings | 25% accuracy collapse on problems solved with 100% solo success; $50M-$200M industry-wide wasted compute estimate |
| Training/Post-Training Lead | RL fixes what SFT cannot because it visits failure states during training | 15-29 pp recoverability gain from GRPO vs 5-7 pp benchmark gain -- a 3-5x multiplier on robustness |

### The overall #1 finding (consensus):

**Optimizing reasoning LLMs for benchmark performance creates an inverse vulnerability: the strongest benchmark models become the most fragile when their reasoning is perturbed by off-distribution content -- the exact condition that occurs in every agentic deployment involving tool use, multi-model collaboration, or external context injection.**

This is the finding that matters to safety teams, engineers, product leaders, academic reviewers, and training researchers simultaneously. It is surprising, well-supported by data (15 models, 9 benchmarks, ablation studies, control experiments), and has immediate practical implications.

---

## Part 3: Rebranding Recommendations (High-Level Narrative)

### What the abstract should lead with:

The abstract should open with the empirical discovery, not the collaborative reasoning vision. Current opening: "Reasoning LLMs are trained to verbalize their thinking process... This transparency also opens a promising direction: multiple reasoners should directly collaborate..." This is aspirational framing for an empirical paper.

**Proposed abstract structure:**
1. **Sentence 1-2:** Reasoning LLMs are increasingly deployed in settings where their thinking is perturbed by external content -- tool outputs, collaborator reasoning, user feedback. We study whether current models can handle this.
2. **Sentence 3:** We introduce "off-trajectory reasoning" and the twin tests (recoverability and guidability).
3. **Sentence 4-5:** The headline findings: (a) stronger models are MORE fragile under perturbation (the robustness-performance tradeoff), with AM-Thinking-32B dropping from 100% to 33.4% on problems it reliably solves; (b) no model can leverage correct guidance beyond its capability boundary (<9.2%).
4. **Sentence 6:** The training insight: teacher vulnerabilities transfer through distillation even on correct-only data; RL closes the gap.
5. **Sentence 7:** Implications for multi-model collaboration, agentic safety, and training pipelines.

### What the headline finding should be:

"Benchmark-optimized reasoning LLMs are inversely robust: the highest-scoring models on standard benchmarks exhibit the largest performance collapse when their reasoning traces are perturbed by off-distribution content."

This is crisp, counterintuitive, and memorable. It is the finding that gets cited.

### What the intro should motivate:

The introduction currently spends significant space on the collaborative reasoning vision (efficiency, exploration, safety) before arriving at the evaluation framework. The revised intro should:

1. **Open with the practical reality:** Reasoning LLMs in deployment constantly encounter off-distribution content in their reasoning traces -- tool outputs, search results, collaborator reasoning, error messages. This is not a hypothetical; it is the default mode of agentic AI.
2. **Pose the question sharply:** Do current reasoning models handle this well? The implicit assumption is that "stronger" models handle it better. We test this assumption.
3. **Preview the surprising answer:** No. Stronger models are more fragile, not less. And no model can build on correct external reasoning beyond its capability boundary.
4. **Then** present the twin tests framework as the methodology that reveals these findings.
5. **Then** briefly mention the collaborative reasoning vision as the long-term motivation and application domain.

The key structural change: **findings-first, vision-second.** The current paper does vision-first, findings-second. Reversing this order makes the paper immediately compelling rather than requiring the reader to trust a speculative framing before encountering the payoff.

### How findings should be ordered and framed:

**Finding 1 (headline):** The robustness-performance tradeoff. Benchmark-optimized models are inversely robust to off-trajectory perturbation. Frame this with the specific AM-Thinking vs Qwen3-1.7B contrast and the rank-change analysis.

**Finding 2 (the wall):** Guidability failure. No model exceeds 9.2% guidability on shared subsets. Even when 18.6% of steers contain the correct answer, models reject it. This is not just "they can't collaborate" -- it is "they actively override correct external reasoning with their own wrong reasoning."

**Finding 3 (the training insight):** Post-training decisions have lasting, sometimes hidden effects on robustness. Three sub-findings: (a) teacher vulnerability transfer through correct-only data, (b) RL bridges the gap that SFT cannot, (c) aggressive data filtering (LIMO-style) creates high variance in robustness.

**Finding 4 (the ablation insight):** The opening of reasoning is disproportionately important. Distraction at 0% causes the largest drop; preserving the first paragraph (problem restatement) dramatically improves recovery. This is both mechanistically interesting and immediately actionable for tool-use pipeline design.

### What belongs in conclusion/future work:

- The collaborative reasoning vision (currently in the intro) should be moved to future work as the long-term application of these findings.
- Explicit connection to agentic AI safety: "Our recoverability test directly models the threat surface of indirect prompt injection in agentic systems."
- Acknowledgment that results may not generalize to closed-source frontier models (o3, Claude, Gemini).
- Call for adversarial versions of the recoverability test: "Our distractors are benign; adversarially crafted perturbations would likely produce worse results."
- Extension to safety-specific reasoning domains (not just math/coding).
- Multi-turn, multi-injection settings (compounding vulnerability).

---

## Part 4: Three Headline Findings (Exact One-Sentence Formulations)

Synthesized across all 5 perspectives, the three crispest findings:

### Finding 1: The Robustness-Performance Tradeoff

> "Among 15 reasoning LLMs, benchmark-optimized models are inversely robust to off-trajectory perturbation: the highest-scoring model (AM-Thinking-32B, 82.6% avg) recovers from mid-reasoning distractions only 33.4% of the time, while the lowest-tier model (Qwen3-1.7B, 59.9% avg) recovers 98.4% of the time."

### Finding 2: The Guidability Wall

> "No reasoning LLM can leverage correct partial reasoning from a stronger collaborator to solve problems beyond its solo capability: guidability on the shared evaluation subset caps at 9.2%, even when 18.6% of guiding steers already contain the correct final answer."

### Finding 3: Hidden Vulnerability Transfer Through Distillation

> "Student models inherit their teacher's reasoning fragility through distillation even when trained exclusively on correct trajectories, meaning that data quality alone is insufficient -- the teacher's latent reasoning style propagates behavioral vulnerabilities invisible in the training data."

---

## Part 5: What the Paper Should NOT Do

### Where agents disagree:

1. **How strongly to frame safety implications.** The Safety Researcher wants explicit safety framing ("this is an empirical instance of inverse scaling"). The Academic Reviewer cautions that the paper tests math/coding, not safety-specific reasoning, and any safety claim requires a speculation flag. **Resolution:** The paper should make the connection to agentic safety explicit but mark the extension to adversarial settings and safety-specific reasoning as "well-motivated but unverified."

2. **Whether to pitch as a benchmark vs. a methodology.** The Academic Reviewer notes that a benchmark (downloadable, standardized) would drive citations more than a protocol (requires re-implementation). The Training Lead values it primarily as a training diagnostic. **Resolution:** Do not overclaim benchmarking status. Call it an evaluation framework/protocol. Release code for easy adoption.

3. **Dollar-value estimates.** The Product Lead provides specific cost impact numbers ($50M-$200M wasted compute). These should NOT appear in the paper -- they are useful for blog posts and talks but are speculative in an academic context.

### Overclaims to avoid:

- **Do NOT claim this paper demonstrates collaborative reasoning.** It demonstrates that collaborative reasoning is currently broken and identifies why. This is a negative result dressed as a positive contribution -- and that is fine, but the framing must be honest about it.

- **Do NOT claim the inverse capability-robustness relationship is causal.** The paper documents a strong correlation with suggestive control studies, but does not establish the causal mechanism. Saying "benchmark optimization causes fragility" is stronger than the data supports. Saying "benchmark optimization and fragility co-occur, and our control studies suggest training choices as a mediating factor" is precise.

- **Do NOT generalize to closed-source frontier models.** The paper evaluates open-weight models up to 32B. Whether o3, Claude with extended thinking, or Gemini 2.5 Pro exhibit the same inverse scaling pattern is an open question. The Academic Reviewer and Safety Researcher both flag this.

- **Do NOT frame benign distractors as adversarial robustness results.** The distractors are coherent but contextually irrelevant reasoning snippets from the same model. They are not adversarially optimized. The 25% failure rate is almost certainly a lower bound on adversarial vulnerability, but claiming adversarial robustness implications without adversarial experiments is overreach.

- **Do NOT claim the guidability failure means models "cannot reason about external input."** Models process external input fine in other contexts (RAG, few-shot examples). The failure is specifically about building on off-distribution partial reasoning traces to solve problems beyond the model's capability boundary.

### Framing that would be counterproductive:

- Framing the paper primarily as "why multi-model collaboration fails" -- this makes the paper sound like a dead end rather than a diagnostic tool.
- Over-emphasizing the LIMO critique -- LIMO is one data point and the authors of LIMO could reasonably object that the comparison is unfair (different training data scale, different objectives).
- Marketing the paper as a "safety paper" when it evaluates math/coding. It has safety implications, which should be stated clearly, but it is fundamentally a reasoning evaluation paper.

---

## Part 6: Highest-Leverage Actions (Ranked by Impact-to-Effort Ratio)

### Action 1: Restructure the abstract to lead with findings (Impact: 10/10, Effort: 2/10)

**What to change:** Rewrite the abstract to open with the empirical discovery (robustness-performance tradeoff), not the collaborative reasoning vision. Move the vision to sentence 5-6. Add the teacher vulnerability transfer finding.

**Why highest leverage:** The abstract is what 90% of readers will see. It determines whether they read the paper. Every agent independently said the abstract buries the lead. This is a 200-word text change that transforms the paper's first impression.

**Specific edit:** Replace the first two sentences ("Reasoning LLMs are trained to verbalize... yielding strong gains" and "This transparency also opens a promising direction...") with something like: "We discover a robustness-performance tradeoff in reasoning LLMs: models optimized for benchmark accuracy become more vulnerable to perturbation of their reasoning traces, not less." Then introduce the twin tests framework and key numbers.

### Action 2: Add one sentence connecting recoverability to agentic tool use (Impact: 8/10, Effort: 1/10)

**What to change:** In the introduction or discussion, add a single explicit sentence: "The recoverability test directly models a common pattern in agentic AI deployment: when tool outputs (search results, code execution, API responses) are injected into a model's reasoning context, they constitute off-trajectory content whose effect on reasoning quality has not been systematically evaluated."

**Why high leverage:** This one sentence connects the paper to the largest and fastest-growing area of LLM deployment. It makes the paper relevant to every team building agents, coding assistants, or tool-augmented systems. Currently this connection requires the reader to make the inference themselves; making it explicit costs nothing and dramatically broadens the audience.

### Action 3: Promote Finding 3 (teacher vulnerability transfer) to abstract-level prominence (Impact: 8/10, Effort: 2/10)

**What to change:** The abstract currently mentions this finding ("sub-optimal recoverability behaviors of teacher models are transferred to distilled students even if the distilled data trajectories are correct") but it is buried in sentence 8 of 9. Move it up. Give it a crisp formulation alongside the other headline findings.

**Why high leverage:** All five agents identified this as the paper's most novel training insight. It is actionable (it changes how you select distillation teachers), surprising (correct-only data still transmits vulnerabilities), and mechanistically interesting. Elevating it costs one sentence in the abstract.

### Action 4: Reframe "off-trajectory reasoning" as a measurable property, not just a setting (Impact: 7/10, Effort: 2/10)

**What to change:** Throughout the paper, shift language from "we study off-trajectory reasoning" to "we measure a model's off-trajectory robustness." Introduce phrases like "off-trajectory robustness score" or "recoverability score" that people can naturally use to compare models.

**Why high leverage:** As the Academic Reviewer notes, "A property is more citeable than a setting." If practitioners can say "Model X has 33% off-trajectory robustness vs Model Y at 98%," the paper's vocabulary enters the discourse naturally. This is primarily a find-and-replace level change in framing language.

### Action 5: Add a paragraph in Discussion/Conclusion explicitly connecting findings to the broader "does safety scale with capability?" question (Impact: 7/10, Effort: 3/10)

**What to change:** Add 4-6 sentences in the discussion connecting the inverse capability-robustness finding to the scaling laws and inverse scaling literatures. Reference McKenzie et al. (2023) on inverse scaling. Note that the paper's results suggest a specific instance where capability gains on benchmarks do not translate to (and may inversely correlate with) robustness -- a finding relevant to the "does safety scale?" question in alignment research.

**Why high leverage:** This positions the paper in a debate that the entire AI safety community is watching. The connection is well-supported by the data (15 models, clear inverse trend) and requires only a brief discussion paragraph, not new experiments. It dramatically increases the paper's relevance to a high-attention audience without making unsupported claims (the qualifier "in the specific domain of reasoning trace perturbation" keeps it honest).

---

## Summary: The Rebranding in One Paragraph

This paper currently tells the story: "We wanted to study collaborative reasoning, so we built twin tests, and we found some surprising things." It should instead tell the story: "We discovered that the strongest reasoning LLMs are the most fragile when their reasoning is perturbed -- a finding with immediate implications for agentic AI, tool use, and multi-model systems. We built a principled evaluation framework (twin tests) that reveals this hidden vulnerability, and we traced its origins to specific training decisions: distillation teacher choice transmits hidden fragility even through correct data, while RL training provides a fix that SFT cannot. These findings redefine what it means for a reasoning model to be 'strong' and provide the first evaluation infrastructure for a dimension of reasoning quality that standard benchmarks entirely miss."

The data does not change. The experiments do not change. The story changes from "here is a framework for studying collaboration" to "here is a discovery about reasoning fragility, with a framework for measuring it and training insights for fixing it." The former is a methodology paper. The latter is a findings paper. The findings are strong enough to lead.
