# Perturb-R (Off-Trajectory Reasoning)

> Evaluating and analyzing LLM reasoning robustness under off-trajectory perturbations (distractors, recoverability, etc.).

## Current TODOs
- [ ] Writing polish: "frontier open-weight LLMs" → "representative" (line ~100)
- [ ] Writing polish: AI usage statement cleanup
- [ ] Table 1/2 captions: document that shading intensity = rank change magnitude
- [ ] Appendix caption period consistency (several missing trailing periods)
- [ ] Conclusion expansion: add limitations + future work (2 sentences)
- [ ] Methodology section plain-English overhaul (reviewer promise)
- [ ] Attention-based mechanistic analysis in appendix (reviewer promise)
- [ ] Add likelihood-based probe for guidability to paper (done in rebuttal, not in paper)
- [ ] Add qualitative guidability examples to paper (done in rebuttal, not in paper)
- [ ] Add digit-corruption experiment to paper (done in rebuttal, not in paper)
- [ ] Introduction Steps 1-6: language tightening + P5 rewrite + P6 lead-in (from plan)

## In Progress
- Camera-ready polish (ICLR 2026) — Batches 1-4 partially done

## Recently Done
- [2026-03-02] Section 3: add coding recoverability example to Finding 1 (OpenThinker3 vs R1-Qwen-32B)
- [2026-03-02] Section 3: add coding guidability sentence to "Even correct guidance fails"
- [2026-03-02] Fix R1-Llama-8B RecInd delta: +2 → +1 (tie with DeepScaleR at 46.2)
- [2026-03-02] Reorder appendix coding guidability table by benchmark avg
- [2026-03-02] Fix R1-Qwen-32B rank deltas for bench tie with AM-Thinking (89.7)
- [2026-03-02] Fix coding Table 2: switch recoverability to distractor_ratio=0.2 (matching math)
- [2026-03-02] Fix coding tables: switch to micro-averaging (matching math)
- [2026-03-02] Appendix coding guidability table: update from macro to micro numbers
- [2026-03-02] Subagent verification: all 92 rank deltas in Tables 1 & 2 confirmed correct
- [2026-03-02] Full revision audit across 3 planning docs (10 done, 5 partial, 10 remaining)

## Key Decisions
- **Micro-averaging everywhere**: Both math and coding tables use micro-averaging for recov/guidability metrics. Benchmark Avg stays macro (standard convention).
- **distractor_ratio=0.2 for recoverability**: Both math and coding use only r=0.2 (easiest perturbation level) for the main recoverability numbers.
- **Competition ranking for ties**: `scipy.stats.rankdata(method='min')` — tied models get the same rank.
- **Rank delta formula**: `metric_rank - bench_rank` with ascending ranks (1 = lowest score).
- **Consistent terminology**: Use "steers already containing the correct answer" (not "answer-forcing" or "leakage") for guidability analysis.
- **Title**: "on Reasoning Trajectories?" (plural)

## Known Issues / Blockers
- uv setup not ready — using conda instead
- Coding guidability only computed for 7 models (1.5B-8B tier); 32B models missing
