# Camera-Ready Revision Plan: Off-Trajectory Reasoning (ICLR 2026)

**Paper:** `docs/iclr-paper/iclr2026/iclr2026_conference.tex`
**Date:** 2026-02-27
**Status:** Accepted at ICLR 2026, camera-ready stage

This document consolidates all revision suggestions from:
- **Round 1:** 10 brainstorming agents (interleaved thinking, adversarial tool injection, model handoff, agentic AI framing, counterintuitive narrative, RL training insights, abstract/intro rewrites, conclusion drafts)
- **Round 2:** 5 expert persona agents (ML Safety Researcher, Industry ML Engineer, Senior Academic Reviewer, AI Product/Startup Lead, Training/Post-Training Lead)
- **Synthesis agent:** Unified rebranding strategy distilled from all 15 agents

---

## Table of Contents

1. [Executive Summary: The Rebranding Thesis](#1-executive-summary)
2. [Three Headline Findings](#2-three-headline-findings)
3. [Narrative Strategy: What Changes](#3-narrative-strategy)
4. [Concrete Revision Actions (Ranked)](#4-concrete-revision-actions)
5. [Abstract Revision](#5-abstract-revision)
6. [Introduction Revision](#6-introduction-revision)
7. [Findings Reordering](#7-findings-reordering)
8. [Conclusion & Future Work Expansion](#8-conclusion-and-future-work)
9. [Writing & LaTeX Fixes (Original Batches)](#9-writing-and-latex-fixes)
10. [Guardrails: What NOT to Do](#10-guardrails)
11. [Supporting Research & Citations](#11-supporting-research)

---

## 1. Executive Summary

### The Core Problem

The paper currently tells the story: *"We wanted to study collaborative reasoning, so we built twin tests, and we found some surprising things."*

### The Rebranded Story

It should tell: *"We discovered that the strongest reasoning LLMs are the most fragile when their reasoning is perturbed -- a finding with immediate implications for agentic AI, tool use, and multi-model systems. We built a principled evaluation framework (twin tests) that reveals this hidden vulnerability, and traced its origins to specific training decisions: distillation teacher choice transmits hidden fragility even through correct data, while RL training provides a fix that SFT cannot."*

### Why This Matters

All 15 agents across both rounds converge on five unanimous critiques:

1. **The headline finding is buried.** The inverse capability-robustness relationship (AM-Thinking-32B: 82.6% benchmark, 33.4% recoverability vs Qwen3-1.7B: 59.9% benchmark, 98.4% recoverability) is the paper's most important result, but the abstract/intro lead with the collaborative reasoning vision.

2. **The teacher vulnerability transfer finding is undersold.** Students inherit teacher fragility through correct-only data -- this is mechanistically novel, practically actionable, and belongs at abstract-level prominence.

3. **The framing promises "collaborative reasoning" but delivers "reasoning robustness."** Rebalance: lead with what the paper proves (robustness findings), motivate with where it points (collaboration prerequisites).

4. **The agentic AI connection is underexploited.** Every tool call = off-trajectory injection. Recoverability directly measures agentic robustness. Currently a footnote; should be explicit and prominent.

5. **RL vs SFT finding is actionable and should be elevated.** 15-29 pp recoverability gain from GRPO vs 5-7 pp benchmark gain. "SFT teaches what good reasoning looks like; RL teaches what to do when reasoning goes bad."

---

## 2. Three Headline Findings

Synthesized from all 15 agents, these are the crispest one-sentence formulations:

### Finding 1: The Robustness-Performance Tradeoff

> "Among 15 reasoning LLMs, benchmark-optimized models are inversely robust to off-trajectory perturbation: the highest-scoring model (AM-Thinking-32B, 82.6% avg) recovers from mid-reasoning distractions only 33.4% of the time, while the lowest-tier model (Qwen3-1.7B, 59.9% avg) recovers 98.4% of the time."

### Finding 2: The Guidability Wall

> "No reasoning LLM can leverage correct partial reasoning from a stronger collaborator to solve problems beyond its solo capability: guidability on the shared evaluation subset caps at 9.2%, even when 18.6% of guiding steers already contain the correct final answer."

### Finding 3: Hidden Vulnerability Transfer Through Distillation

> "Student models inherit their teacher's reasoning fragility through distillation even when trained exclusively on correct trajectories, meaning that data quality alone is insufficient -- the teacher's latent reasoning style propagates behavioral vulnerabilities invisible in the training data."

---

## 3. Narrative Strategy

### Current Structure (vision-first)
1. Intro: Collaborative reasoning vision (efficiency, exploration, safety)
2. Methods: Twin tests framework
3. Findings: Results presented as observations
4. Control study: Training analysis
5. Conclusion: Brief summary

### Proposed Structure (findings-first)
1. Intro: Practical reality (agentic AI, tool use) → sharp question → surprising answer → framework
2. Methods: Twin tests as the diagnostic tool
3. Findings: Reordered by impact (see Section 7)
4. Control study: Training insights elevated
5. Conclusion: Expanded with implications, limitations, future work

### Key Structural Change

**Findings-first, vision-second.** The current paper does vision-first, findings-second. Reversing this order makes the paper immediately compelling rather than requiring the reader to trust a speculative framing before encountering the payoff.

---

## 4. Concrete Revision Actions (Ranked by Impact/Effort)

| Priority | Action | Impact | Effort | Section |
|----------|--------|--------|--------|---------|
| **1** | Restructure abstract to lead with findings | 10/10 | 2/10 | [5](#5-abstract-revision) |
| **2** | Add 1 sentence connecting recoverability to agentic tool use | 8/10 | 1/10 | [6](#6-introduction-revision) |
| **3** | Promote teacher vulnerability transfer to abstract prominence | 8/10 | 2/10 | [5](#5-abstract-revision) |
| **4** | Reframe "off-trajectory reasoning" as a measurable *property* | 7/10 | 2/10 | Throughout |
| **5** | Add discussion paragraph on "does safety scale with capability?" | 7/10 | 3/10 | [8](#8-conclusion-and-future-work) |
| **6** | Expand conclusion with limitations + future work | 6/10 | 3/10 | [8](#8-conclusion-and-future-work) |
| **7** | Elevate Finding 3b (correct answer in guidance, still fails) | 6/10 | 2/10 | [7](#7-findings-reordering) |
| **8** | Apply remaining writing fixes (Batches 3-5) | 5/10 | 3/10 | [9](#9-writing-and-latex-fixes) |
| **9** | Add 1-2 intro sentences on model handoff/guidability motivation | 5/10 | 1/10 | [6](#6-introduction-revision) |
| **10** | Sharpen opening line of introduction | 4/10 | 1/10 | [6](#6-introduction-revision) |

---

## 5. Abstract Revision

### Current Abstract Structure
1. Reasoning LLMs are trained to verbalize thinking → strong gains
2. Transparency opens collaborative reasoning direction
3. We introduce off-trajectory reasoning
4. Twin tests: recoverability and guidability
5. Evaluate 15 models on 9 benchmarks
6. Finding: benchmark ≠ off-trajectory robustness; sometimes inversely correlated
7. Guidability <10%
8. Teacher vulnerabilities transfer through distillation (buried, sentence 8 of 9)
9. RL control study insights

### Proposed Abstract Structure
1. **Lead with the practical setting + finding:** Reasoning LLMs in agentic deployment encounter off-distribution content in their reasoning traces (tool outputs, collaborator reasoning). We test whether stronger models handle this better. They don't.
2. **Introduce the framework:** We formalize this as "off-trajectory reasoning" via twin tests: recoverability (resist distracting steers) and guidability (leverage correct guidance).
3. **Headline numbers:** AM-Thinking-32B (82.6% benchmark) recovers only 33.4%; Qwen3-1.7B (59.9% benchmark) recovers 98.4%. Guidability caps at 9.2%.
4. **Training insight elevated:** Teacher vulnerabilities transfer through distillation on correct-only data; RL (GRPO) provides 15-29 pp recoverability gains vs 5-7 pp benchmark gains.
5. **Implication:** These findings redefine what "strong reasoning" means for agentic and multi-model settings.

### Draft Abstract Options (from Round 1 agents)

**Version A: Practical Motivation Lead**
> Reasoning LLMs are increasingly deployed in agentic settings where tool outputs, search results, and collaborator reasoning inject off-distribution content into ongoing reasoning traces. We introduce off-trajectory evaluation -- testing whether LLMs can recover from misleading partial reasoning (recoverability) or follow correct guidance from another model (guidability). Evaluating 15 models across 9 math and coding benchmarks, we find that benchmark performance does not predict -- and in several cases inversely correlates with -- off-trajectory robustness. The highest-scoring model recovers from mid-reasoning distractions only 33.4% of the time, while a model scoring 23 points lower on benchmarks recovers 98.4%. No model exceeds 9.2% guidability on shared problems, even when the correct answer is present in the guidance. We trace this fragility to training methodology: RL (GRPO) yields 15-29 pp recoverability gains, while teacher vulnerabilities transfer to distilled students even through correct-only training data. These results have direct implications for agentic AI safety, model cascading, and training pipeline design.

**Version B: Counterintuitive Finding Lead**
> Models that excel at solo reasoning are not necessarily good at reasoning with others. We introduce off-trajectory evaluation -- testing whether LLMs can recover from misleading partial reasoning (recoverability) or follow correct guidance from a collaborator (guidability). Evaluating 15 models across 9 math and coding benchmarks, we find that standard benchmark performance does not predict -- and in several cases inversely correlates with -- off-trajectory robustness. We trace this fragility to training methodology: reinforcement learning (GRPO) yields 15-29 percentage-point recoverability gains, while supervised fine-tuning on curated data produces high-scoring but brittle reasoners. These results have direct implications for agentic AI, where every tool call injects out-of-distribution content into an ongoing reasoning process, and for model cascading, where our guidability results show that mid-reasoning handoff between models remains an unsolved problem.

**Version C: Evaluation Gap Lead**
> Standard benchmarks evaluate reasoning LLMs in isolation, but deployment increasingly requires models to reason alongside tool outputs, collaborator reasoning, and external context. We introduce off-trajectory evaluation to measure this missing dimension. Our twin tests -- recoverability and guidability -- reveal that benchmark-optimized models are inversely robust: the highest-scoring model shows only 33.4% recoverability, while the strongest recoverer scores 23 points lower on benchmarks. Guidability is near-zero (<9.2%) across all models. Control experiments identify the mechanism: distillation transfers teacher fragility even through correct data, while RL training bridges the robustness gap. These findings suggest that current training paradigms optimize for the wrong objective when the target deployment involves multi-model or tool-augmented reasoning.

### Recommendation
Round 1 agents recommend **Version B** for its clarity and hook. Round 2 synthesis recommends a hybrid of **A and B**: lead with the practical setting (agentic AI is real, not speculative), follow with the counterintuitive finding. The key is that the *finding* comes in the first 2 sentences, not the *framework*.

---

## 6. Introduction Revision

### Current Intro Flow
1. Reasoning LLMs verbalize thinking → strong gains
2. Collaborative reasoning vision (efficiency, exploration, safety)
3. Challenge: distribution shift when reasoning on others' trajectories
4. We study this via twin tests
5. Key findings listed
6. Contributions listed

### Proposed Intro Flow
1. **Open with practical reality:** Reasoning LLMs in deployment encounter off-distribution content constantly -- tool outputs, search results, collaborator reasoning, error messages. This is the default mode of agentic AI.
2. **Pose the question sharply:** Do current reasoning models handle this well? The implicit assumption is that "stronger" models handle it better.
3. **Preview the surprising answer:** No. Stronger models are more fragile, not less. And no model can build on correct external reasoning beyond its capability boundary.
4. **Present the framework:** We formalize this as "off-trajectory reasoning" via twin tests.
5. **Brief mention of collaborative reasoning vision** as the long-term motivation.
6. **Contributions** (unchanged).

### Specific Sentences to Add

**Agentic AI connection (Action #2):**
> "The recoverability test directly models a common pattern in agentic AI deployment: when tool outputs (search results, code execution, API responses) are injected into a model's reasoning context, they constitute off-trajectory content whose effect on reasoning quality has not been systematically evaluated."

**Model handoff connection:**
> "Similarly, guidability measures whether partial reasoning from one model can be productively continued by another -- a prerequisite for model cascading under cost or rate constraints."

### Opening Line Candidates (from Round 1 agents)

- *"The reasoning models that score highest on benchmarks may be the worst collaborators."*
- *"We show that the very training recipe that produces state-of-the-art solo reasoners actively undermines their ability to reason with others."*
- *"Standard benchmarks evaluate reasoning LLMs in isolation, but deployment increasingly requires models to reason alongside tool outputs, collaborator reasoning, and external context."*

---

## 7. Findings Reordering

### Current Finding Order
1. Benchmark performance ≠ off-trajectory robustness (largely orthogonal)
2. Guidability <10%
3. Training insights (distillation, RL)

### Proposed Finding Order

**Finding 1 (headline): The Robustness-Performance Tradeoff**
- Benchmark-optimized models are inversely robust to off-trajectory perturbation
- Frame with AM-Thinking-32B vs Qwen3-1.7B contrast
- Include the rank-change analysis

**Finding 2 (the wall): The Guidability Wall**
- No model exceeds 9.2% guidability on shared subsets
- **Elevate sub-finding 3b:** Even when 18.6% of steers contain the correct answer, models reject it
- Give Finding 3b a bold sub-heading: **"Even Correct Guidance Fails"**
- Frame: "models actively override correct external reasoning with their own wrong reasoning"

**Finding 3 (training insight): Why This Happens**
- (a) Teacher vulnerability transfer through correct-only data
- (b) RL bridges the gap (15-29 pp recoverability vs 5-7 pp benchmark)
- (c) Aggressive data filtering (LIMO-style) creates high variance in robustness
- Frame the RL insight: "SFT teaches what good reasoning looks like; RL teaches what to do when reasoning goes bad"

**Finding 4 (ablation): The Opening Matters**
- Distraction at 0% causes the largest drop
- Preserving the first paragraph improves recovery by 15-24 pp for the worst models
- Actionable: tool-use pipelines should preserve the model's initial problem framing

---

## 8. Conclusion and Future Work

### Current Conclusion
3 sentences. Too thin. Restates findings without implications.

### Proposed Expansion

**Conclusion Version A (Practical Urgency -- recommended by synthesis):**
> As reasoning LLMs are increasingly deployed in agentic pipelines -- processing tool outputs, collaborating with other models, and encountering distribution shifts during inference -- our results reveal a critical gap between benchmark performance and operational robustness. The finding that stronger solo reasoners are often more fragile collaborators suggests that current training regimes optimize for the wrong objective when the target deployment is multi-model or tool-augmented.

**Limitations to acknowledge:**
- Results are on math/coding domains with open-weight models only
- Distractors are benign (not adversarially crafted) -- actual vulnerability is likely worse
- Closed-source frontier models (o3, Claude, Gemini) untested

**Future work to mention:**
- Extension to natural language reasoning and safety-specific domains
- Adversarial versions of the recoverability test
- Whether off-trajectory robustness can be directly targeted during training
- Multi-turn, multi-injection settings (compounding vulnerability)
- Connection to inverse scaling: "Our results suggest a specific instance where capability gains on benchmarks do not translate to -- and may inversely correlate with -- robustness, a finding relevant to broader questions about whether safety scales with capability."

### Discussion Paragraph on Safety Scaling (Action #5)

> Our finding that benchmark-optimized models exhibit inversely correlated off-trajectory robustness connects to the broader question of whether safety properties scale with capability. In the specific domain of reasoning trace perturbation, we observe that training for higher benchmark scores -- whether through data curation (LIMO) or teacher selection (AM-Thinking) -- can actively degrade robustness to the kind of distribution shifts that occur in every agentic deployment. This suggests that standard benchmark evaluation alone is insufficient for assessing model readiness for agentic or collaborative settings, and that off-trajectory robustness should be explicitly targeted during training and evaluation.

---

## 9. Writing and LaTeX Fixes (Original Batches)

### Batch 1: Grammar & LaTeX Bugs -- DONE (committed d59ed9d)
11 fixes applied (title plural, subject-verb agreement, prepositions, emph in math mode, notation consistency, double space, benchmark list grammar).

### Batch 2: Technical Precision -- PARTIALLY DONE (committed 817a969)
- [x] Fix 14: Clarified "mean 25.1% degradation" baseline
- [x] Fix 15: Removed undefined `r^partial` notation
- [x] Fix 16: Added individual subset scope to guidability claim
- [ ] Fix 12: Abstract hedge ("stronger LLMs are often more fragile") -- *deferred, revisit during abstract rewrite*
- [ ] Fix 13: Finding 1 "largely orthogonal" softening -- *deferred, revisit during findings reorder*

### Batch 3: Writing Quality -- NOT YET APPLIED
| # | Line | Issue | Fix |
|---|------|-------|-----|
| 17 | 63 | "exert direct control" redundant | → "could directly intervene in an LLM's ongoing reasoning to steer its thinking" |
| 18 | 67 | Dangling "due to" clause | → "it remains unclear whether solo-reasoning LLMs can effectively leverage partial reasoning trajectories from other collaborators, given the associated distribution shift" |
| 19 | 102 | "frontier open-weight LLMs" — 1.5B aren't frontier | → "representative open-weight reasoning LLMs" |
| 20 | 474 | "massive improvements" — vague | → "substantial recoverability gains (15--29 percentage points)" |
| 21 | 527 | AI usage statement awkward | → "We used AI assistance for language polishing of author-written text." |
| 22 | 80 | "of which collaborative and off-trajectory reasoning is an intrinsic part" | → "such as collaborative and off-trajectory reasoning" |

### Batch 4: Structure & Presentation -- NOT YET APPLIED
| # | Line | Issue | Fix |
|---|------|-------|-----|
| 23 | 503 | `\label{sec:background}` for Related Work | → `\label{sec:related_work}` |
| 24 | 150 | "Datasets and Benchmarks" heading | → "Models, Datasets, and Benchmarks" |
| 25 | 249/357 | Table color encoding undocumented | Add to caption: "Cell shading proportional to rank change magnitude" |
| 26 | — | Table caption period inconsistency | Standardize: all captions end with period |
| 27 | 426-429 | Section 4 transition abrupt | Add motivating question |
| 28 | 511-514 | Conclusion too thin | Expand (see Section 8 above) |

### Batch 5: Discussion Items
- [x] Finding 3b: Elevate to bold sub-heading (author approved)
- [ ] Counter-evidence at 80% position: Brief acknowledgment sentence
- [ ] Abstract restructuring: See Section 5 above
- [ ] Related work gaps: Review citations from agents

---

## 10. Guardrails: What NOT to Do

### Overclaims to Avoid
- **Do NOT claim this paper demonstrates collaborative reasoning.** It demonstrates that collaborative reasoning is currently broken and identifies why.
- **Do NOT claim the inverse capability-robustness relationship is causal.** Say: "benchmark optimization and fragility co-occur, and our control studies suggest training choices as a mediating factor."
- **Do NOT generalize to closed-source frontier models.** The paper evaluates open-weight models up to 32B.
- **Do NOT frame benign distractors as adversarial robustness results.** The 25% failure rate is a lower bound on adversarial vulnerability, but claiming adversarial robustness without adversarial experiments is overreach.
- **Do NOT claim guidability failure means models "cannot reason about external input."** The failure is specifically about building on off-distribution partial reasoning traces beyond the model's capability boundary.

### Counterproductive Framing
- Don't frame primarily as "why multi-model collaboration fails" (sounds like a dead end)
- Don't over-emphasize the LIMO critique (single data point, authors could object)
- Don't market as a "safety paper" when it evaluates math/coding (it has safety *implications*)
- Don't include dollar-value estimates (useful for talks, speculative in paper)

### Disagreements Across Agents (flagged for author judgment)
1. **Safety framing strength:** Safety Researcher wants explicit "inverse scaling" framing; Academic Reviewer cautions it requires a speculation flag for non-safety domains. *Resolution:* Make the connection explicit but mark as "well-motivated but unverified" for adversarial/safety-specific settings.
2. **Benchmark vs methodology pitch:** Academic Reviewer prefers benchmark framing (more citeable); Training Lead values it as training diagnostic. *Resolution:* Call it an "evaluation framework/protocol." Release code.

---

## 11. Supporting Research and Citations

### From Round 1 Agents: Relevant Work for Related Work / Discussion

**Agentic AI / Tool Injection Safety:**
- InjecAgent (Zhan et al., 2024) -- indirect prompt injection through tool results
- CaMeL (Debenedetti et al., 2025) -- causal reasoning defense against tool manipulation
- BIPIA (Yi et al., 2024) -- benchmarking indirect prompt injection in LLM agents
- AgentDojo (Debenedetti et al., 2024) -- evaluation framework for agent tool-use security
- AgenTRIM (2025) -- tool risk mitigation for agentic AI

**Model Routing / Cascading:**
- FrugalGPT (Chen et al., 2023) -- LLM cascade for cost optimization (query-level routing only)
- RouteLLM (Ong et al., 2024) -- learned routers for cost-quality balance (query-level only)
- SplitReason (Akhauri et al., 2025) -- already cited; our guidability results are directly relevant
- COPE (Nguyen et al., 2024) -- collaborative prompting (query-level, no mid-reasoning handoff)

**Reasoning Robustness (concurrent work):**
- "Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?" (von Recum et al., 2026) -- perturbation study, identifies "doubt" as recovery mechanism
- "Robust Answers, Fragile Logic" (Jiang et al., 2025) -- MATCHA framework, models maintain correct answers with inconsistent reasoning
- "Chain-of-Code Collapse" (Roh et al., 2025) -- adversarial prompting collapses code reasoning
- "The Hypocrisy Gap" (Shiromani et al., 2026) -- detecting sycophantic unfaithful CoT via SAEs

**Training Methodology:**
- DAPO (Yu et al., 2025) -- decoupled clip + dynamic sampling, improvement over GRPO
- Dr. GRPO (Liu et al., 2025) -- identifies GRPO's length bias, fixes with improved optimization
- "SFT Memorizes, RL Generalizes" (Chu et al., 2025) -- directly supports our RL > SFT finding
- "Emergent Search and Backtracking in Latent Reasoning Models" (Cui & Ye, 2026) -- models learn recovery via backtracking

**Interleaved Thinking / Tool Use Architecture:**
- No provider injects tool results directly into reasoning token stream -- all use message-level boundaries (Claude extended thinking, DeepSeek `<think>` blocks, OpenAI hidden CoT)
- This is an implicit acknowledgment of the OOD problem our paper quantifies
- Anthropic applies cryptographic signatures to thinking blocks to prevent tampering

### Key Analogies and Framings (from Round 1)

- **"Hothoused prodigy" effect:** Models trained in pristine conditions develop fragile reasoning that crumbles under perturbation
- **"Trajectory groove":** SFT models learn a narrow groove of successful reasoning; any perturbation derails them
- **"SFT teaches what good reasoning looks like; RL teaches what to do when reasoning goes bad"**
- **Karpathy-style tweet:** "The models that score highest on math benchmarks are the worst at reasoning with others."

---

## Revision Execution Order

### Phase 1: Narrative Restructuring (highest leverage)
1. Rewrite abstract (Action #1, #3)
2. Revise introduction opening (Action #2, #10)
3. Add agentic AI connection sentence (Action #2)
4. Add model handoff motivation sentence (Action #9)

### Phase 2: Findings & Discussion
5. Reorder findings presentation (Section 7)
6. Elevate Finding 3b with bold sub-heading
7. Expand conclusion (Action #6, Section 8)
8. Add safety-scaling discussion paragraph (Action #5)

### Phase 3: Polish
9. Apply Batch 3 writing fixes (#17-22)
10. Apply Batch 4 structure fixes (#23-28)
11. Language pass: reframe as "off-trajectory robustness" property (Action #4)
12. Final LaTeX compile and verification

---

## 12. Structural Skeleton (Synthesized from Step 1 Agents)

*Synthesized from 5 structural skeleton agents (Abstract+Intro Architect, Methods+Eval Architect, Control+Related Architect, Conclusion+Narrative Arc Architect, Devil's Advocate) on 2026-02-27.*

### Title
**Keep as-is**: "Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectories?" — all agents agree.

### Abstract: 8 Sentences, Findings-First

| # | Role | Content |
|---|------|---------|
| S1 | Hook | Reasoning LLMs are increasingly deployed where their chain of thought is not theirs alone (tool outputs, collaborator reasoning, model handoff). |
| S2 | Question | We ask: can solo-trained reasoners handle off-distribution content injected into their reasoning traces? |
| S3 | Framework | We formalize this as *off-trajectory reasoning* via twin tests: recoverability (resist misleading steers) and guidability (leverage correct guidance). |
| S4 | Finding 1 | Evaluating 15 models across 9 benchmarks: benchmark performance does not predict off-trajectory robustness — and in several cases, higher-scoring models perform significantly worse. |
| S5 | Finding 2 | Guidability caps at 9.2% on shared problems; models reject correct guidance 18.6% of the time when the answer is already present. |
| S6 | Training insight | Control studies reveal: teacher vulnerabilities transfer through distillation even on correct-only data. |
| S7 | RL insight | RL (GRPO) yields 15–29 pp recoverability gains vs. 5–7 pp benchmark gains. |
| S8 | Implication | These results have implications for agentic deployment, model cascading, and training pipeline design. |

**Devil's Advocate guardrails:** S4 says "does not predict...in several cases" (not "inversely correlates" — driven by 2 outliers). S8 says "implications for" (not "directly models").

### Introduction: 5 Paragraphs, Reality-First

| P# | Role | What it does |
|----|------|-------------|
| P1 | Practical reality | LLMs with thinking abilities are frontier. Deployment increasingly involves off-distribution content in reasoning traces. Keep existing citations. Add 1 sentence on agentic connection as *implication*. |
| P2 | Sharp question + surprising answer | Can solo-trained models handle this? Twin tests preview. Counterintuitive finding previewed. Collaborative vision compressed to 2 sentences (from full paragraph). |
| P3 | Framework description | Twin tests: recoverability and guidability. Wrapfigure stays. Largely unchanged. |
| P4 | Key findings + training insights | Current findings summary tightened. Teacher transfer elevated. |
| P5 | Contributions | Unchanged numbered list. |

### Section 2: Twin Tests — Minimal Changes
- Add ~2 sentences reframing tests as measuring *properties* of off-trajectory robustness.
- Add 1 sentence connecting to agentic deployment (tool outputs as off-trajectory content).
- Keep everything else.

### Section 3: Findings — Reordered

| Current | Proposed | Rationale |
|---------|----------|-----------|
| Finding 1: "Stronger ≠ stronger collaborators" (recov + guid combined) | **Finding 1: The Robustness-Performance Gap** (recoverability ONLY) | Splitting makes each sharper |
| Finding 2: "Beginning of reasoning is critical" | **Finding 2: The Guidability Wall + "Even Correct Guidance Fails"** (elevated 3b with bold subheading) | Guidability wall more impactful |
| Finding 3: "LLMs fail to leverage correct guidance" | **Finding 3: The Opening Matters** (position analysis + ablation) | Still interesting but lower impact |

**Guardrail:** Finding 1 language stays "do not positively correlate" / "largely orthogonal."

### Section 4: Control Study — Small Additions
- Add transition paragraph (2-3 sentences) framing the puzzle.
- Add closing synthesis paragraph (2-3 sentences) after §4.3.
- Keep current subsection titles and structure.

### Section 5: Related Work — Small Additions
- Add von Recum et al. (2026) concurrent citation (1 sentence).
- Add 1 sentence on agentic AI relevance.
- Keep two-paragraph structure.

### Section 6: Conclusion — Expanded to 4 Paragraphs (~250-300 words)

| P# | Role |
|----|------|
| P1 | Findings summary (twin tests, stronger ≠ better, guidability near-zero, correct guidance fails) |
| P2 | Training implications (teacher transfer, RL > SFT, "SFT teaches what good reasoning looks like; RL teaches what to do when reasoning goes bad") |
| P3 | Limitations (math/coding, open-weight, benign distractors) |
| P4 | Future work + closing (NL reasoning, adversarial, direct training, multi-turn, agentic implication) |

**Guardrails:** Keep ~250-300 words. Limitations paragraph mandatory. Avoid "inverse scaling" language.

### Narrative Arc

```
Setup:     LLMs reason in isolation, but deployment puts others' content in their traces
Question:  Do stronger models handle this better?
Surprise:  No — and no model can leverage correct external guidance
Explain:   Training methodology is the key: teacher choice, RL vs SFT, data curation
Imply:     Current training optimizes for the wrong thing when deployment is multi-model
```

### Change Magnitude Summary

| Section | Magnitude | What changes |
|---------|-----------|-------------|
| Abstract | Medium | Reorder sentences, lead with findings not vision |
| Intro | Medium | Compress vision paragraph, add 1-2 agentic sentences, preview findings earlier |
| §2 Twin Tests | Small | Add ~3 sentences for property framing + agentic connection |
| §3 Findings | Medium | Reorder finding paragraphs (no subsection restructuring) |
| §4 Control | Small | Add transition paragraph + closing synthesis (~5 sentences total) |
| §5 Related Work | Small | Add 1 citation + 1 sentence |
| §6 Conclusion | Medium | Expand from 3 sentences to 4 paragraphs |

### Author Decisions (from this session)
- Findings reorder: Approved
- Abstract tone: Cautious ("does not predict...in several cases")
- Agentic framing: 1 sentence each in intro, §2, related work, conclusion — not overclaimed
- Conclusion: ~250-300 words, 4 paragraphs
