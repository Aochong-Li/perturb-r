# Review Gap Analysis: What's Missing from the Camera-Ready

Paper: "Off-Trajectory Reasoning" (ICLR 2026)
OpenReview: https://openreview.net/forum?id=hVUIguIm14
Generated: 2026-02-27

---

## CRITICAL (Promised in Rebuttal — Must Do)

| # | Item | Source | What's Missing |
|---|------|--------|---------------|
| **1** | **Methodology section overhaul** | Y2jS (raised score 4→6 contingent on this) | Authors promised to: (a) add implementation details (coherent prefix truncation, "Wait. Let me think" transition), (b) improve readability with plain English + examples, (c) justify design choices via prior work constraints. **None of this is done.** |
| **2** | **Attention-based mechanistic analysis** | zPNH + AC remained concern | Authors ran Lasso classifiers on attention features (70-80% balanced accuracy), identified ~5 predictive heads per model, computed Cohen's d. Explicitly promised "We will include full results in the updated paper." **Not in paper.** |
| **3** | **Missing citations from rebuttal** | Y2jS | Speculative Thinking (Yang et al., 2025), ReMa (Wan et al., 2025) — **not in bibliography**. s1 (Muennighoff et al., 2025) — in bib but **not cited**. These were used extensively to justify the protocol. |

---

## HIGH PRIORITY (Promised or Strongly Expected)

| # | Item | Source | What's Missing |
|---|------|--------|---------------|
| **4** | **Likelihood-based probe for guidability** | zPNH | Authors ran per-token probability analysis of guidance region, found no consistent difference. Presented in rebuttal but not in paper. |
| **5** | **Qualitative guidability examples** | BaPD | Reviewer asked for success vs. failure cases. Authors provided 2 examples in rebuttal. Not in paper (could be appendix). |
| **6** | **Digit-corruption / logical fallacy experiment** | zPNH Q1 | Authors ran a controlled experiment showing >80% recovery + high correlation with benchmark perf (unlike their distraction test). Interesting contrast but not in paper. |

---

## NICE-TO-HAVE (Suggestions, Not Promised)

| # | Item | Source | What's Missing |
|---|------|--------|---------------|
| **7** | **Realism/limitations discussion** | AC remained concern, BJqx W1 | No explicit discussion of gap between twin-test protocol and real multi-agent collaboration. No mention of multi-turn, iterative exchanges, mixed-quality steers. |
| **8** | **Computational costs discussion** | BaPD | Brief paragraph noting focus isn't efficiency, guide steers are reused, citing SplitReason for cost analysis. |

---

## ALREADY DONE ✓

- Coding benchmarks (Table 2) — all reviewers satisfied
- Cross-family distractor ablation (Appendix table)
- Recoverability-by-solve-rate table (Appendix D)
- "bechmark" typo fix

---

## Recommended Priority Order

1. **Methodology section rewrite** (items 1+3) — highest priority, reviewer Y2jS conditioned score raise on it
2. **Attention analysis** (item 2) — need results ready to add as appendix section + brief main-text mention
3. **Appendix additions** (items 4-6) — quick wins, can go in appendix
4. **Limitations paragraph** (item 7) — a few sentences in conclusion

---

## Detailed Breakdown by Reviewer

### Reviewer Y2jS (Score: 4 → 6)

**What they want in the methodology section:**
- Explain that truncation happens at sentence boundaries (coherent prefix), not arbitrary token indices
- Explain the "Wait. Let me think" transition phrase between original and injected reasoning
- Plain English walk-through of the protocol (like the rebuttal language)
- Justify why direct injection into assistant thinking (not user message) was chosen
- Reference prior work constraints: SplitReason, Speculative Thinking, ReMa, s1, Thinking Intervention

**Author promise (Note [4]):**
> "We will incorporate the reviewer's suggestions to further improve readability. In particular, we will add additional implementation details in the methodology section and better motivate the design choices in our twin-test protocol by explaining the constraints from prior work."

**Status:** None of this has been incorporated.

### Reviewer zPNH (Score: 4, no post-rebuttal response)

**What they want:**
- Mechanistic analysis explaining WHY models fail (attention weights, intermediate step tracing)
- Distinguish "fail to recognize relevance of guidance" vs. "cannot integrate guidance"
- Consider logical fallacy distractors (digit corruption experiment was done but not added)

**Author promises:**
- Attention analysis: "We will include more detailed results in the updated paper"
- Attention heads: "We will include full results to the updated paper"
- Likelihood probe: conducted but not explicitly promised to include

**Status:** None of the attention/mechanistic analysis is in the paper.

### Reviewer BaPD (Score: 6, no post-rebuttal response)

**What they want:**
- Qualitative examples of successful vs. unsuccessful guidance integration
- Discussion of computational costs

**Status:**
- Qualitative examples: provided in rebuttal, not in paper
- Computational costs: not discussed (low priority — reviewer gave 6 already)

### Reviewer BJqx (Score: 6, no post-rebuttal response)

**What they want:**
- Discussion of multi-turn extension and realistic collaboration dynamics
- Acknowledgment of limitations (one-shot injection vs. iterative back-and-forth)

**Status:** Not discussed. Connects to AC's "remained concern" about realism.

### Meta-Review / AC

**Remained concerns (accepted despite these):**
1. Realism of benchmark for real-world collaboration — "only partially addressed"
2. Shallow mechanistic analysis — "remains limited"

**Expectation:** Authors fulfill their rebuttal promises in the camera-ready.

---

## Post Camera-Ready: Dissemination Plan

### 1. Project Website
- Interactive visualizations of key results (recoverability/guidability heatmaps, rank-change tables, position analysis curves)
- Model comparison explorer — let visitors pick models and see head-to-head off-trajectory performance
- Animated figure showing the twin-test protocol (solo → distracted → guided)
- Embed paper PDF + link to OpenReview, code, and data

### 2. Blog Post (Substack)
- Long-form narrative version of the paper for general ML audience
- Lead with the counterintuitive hook: "the best reasoning model is the most fragile"
- Include interactive/animated versions of key figures
- End with implications for multi-agent systems and AI safety

### 3. Twitter/X Thread
- Visual thread (8-10 tweets) with key figures and one-liner findings
- Tweet 1: Hook with AM-Thinking vs Qwen3 contrast
- Tweet 2-3: What are recoverability and guidability (twin-test figure)
- Tweet 4-5: Key findings with table screenshots
- Tweet 6-7: Control study insights (teacher transfer, RL gains)
- Tweet 8: Link to paper, website, blog post
- Tag relevant accounts (model authors, benchmark creators)
