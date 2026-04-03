# Workshop Paper Draft — Status & Structure

**Working Title:** Projection-Based Distance Filtering for Accelerating NN-Descent Graph Construction
**Author:** Jainish Pandya, University of Waterloo
**Target:** 8 pages + references
**Format:** IEEE (two-column)

---

## Page Budget

| Section | Pages | Status |
|---------|-------|--------|
| Abstract | 0.25 | TO WRITE |
| 1. Introduction | 1.25 | TO WRITE (material ready from survey) |
| 2. Related Work & Background | 1.5 | TO WRITE (can adapt from survey) |
| 3. Method | 1.5 | TO WRITE (code + math exists) |
| 4. Experiments & Results | 2.5 | PARTIALLY DONE (results exist, need more runs) |
| 5. Discussion | 0.5 | TO WRITE |
| 6. Conclusion & Future Work | 0.5 | TO WRITE |
| References | ~1 page | MOSTLY READY (from survey) |

---

## 1. Introduction (~1.25 pages)

### Story Arc
> ANN is critical → field evolved through hashing/trees/quantization/graphs → now hybrids →
> NN-Descent still matters (CAGRA, parallelizability) → its bottleneck is wasted distance
> computation → we propose projection-based filtering to fix it.

### Paragraph-by-Paragraph Plan

**Para 1 — ANN matters now more than ever** (adapt from survey intro)
- LLMs, RAG, vector databases → ANN is infrastructure
- Exact search is O(nd), prohibitive at scale
- STATUS: Can adapt directly from survey paper Section I

**Para 2 — Evolutionary view (CONDENSED to one paragraph)**
- Hashing (LSH, 1999) → Trees (KD-tree, RP-tree) → Quantization (PQ) → Graphs (NSW → HNSW)
- Key insight: field moved from "search-only" structures to "build expensive, search fast" (graphs)
- Timeline: LSH (1999) → PQ (2011) → NN-Descent (2011) → HNSW (2018/2020)
- STATUS: Material exists in survey Sections III-IV. Need to CONDENSE to ~6-8 sentences.

**Para 3 — Hybrid era: combining strengths**
- PyNNDescent = NN-Descent + RP-trees (init)
- LSH-APG = LSH + proximity graph construction
- FINGER = low-rank projections + HNSW search
- SHG = shortcuts + compressed upper layers
- Key pattern: lightweight structures (hashing, projections) used to FILTER or ACCELERATE graph operations
- STATUS: Material exists in survey Section V. Need to condense.

**Para 4 — Why NN-Descent still matters**
- HNSW dominates search, but is CPU-bound and hard to parallelize
- NN-Descent is the backbone of CAGRA (NVIDIA GPU graph construction)
- Also used in EFANNA, PyNNDescent
- Simple, embarrassingly parallel local-join structure
- Yet: its main bottleneck is WASTED distance computations during construction
- STATUS: NEED to gather specific citations for CAGRA using NN-Descent.
  The claim about HNSW being CPU-limited needs a citation or careful phrasing.

**Para 5 — The waste problem + our contribution**
- [FIGURE 1: "Waste ratio" — what fraction of distance computations per iteration DON'T update the graph]
- This motivates: can we SKIP these wasted computations cheaply?
- Our contribution: adapt projection-based distance filtering (from LSH-APG) to NN-Descent construction
- First application of this filtering to NN-Descent
- 1.1–1.4x wall-clock speedup on high-dimensional data, <3% recall loss at conservative settings
- STATUS: NEED to build the waste-ratio plot from existing iteration stats.
  Contribution statement is ready.

### What's READY for Intro
- [x] Evolutionary narrative (from survey)
- [x] Hybrid algorithms discussion (from survey Section V)
- [x] Contribution statement

### What's NEEDED for Intro
- [ ] "Waste ratio" figure (measure update_rate per iteration from existing results)
- [ ] CAGRA citation / verification that it uses NN-Descent
- [ ] Careful claim about HNSW parallelization limitations

---

## 2. Related Work & Background (~1.5 pages)

### 2.1 NN-Descent Algorithm (~0.5 page)
- Algorithm overview: random init → local join → new/old partition → convergence
- Complexity: O(n^1.14) empirically (Dong et al. 2011)
- Key parameters: k (neighbors), rho (sampling), delta (convergence)
- The "local join" is where all distance computations happen
- STATUS: Can write from code + Dong et al. paper. Straightforward.

### 2.2 Filtering in Graph-Based ANN (~0.5 page)
- **FINGER** (Chen et al. 2023): low-rank projections for HNSW *search* — 20-60% speedup
- **LSH-APG** (Zhang et al. 2023): LSH for graph *construction* + search entry points
  - Key: they use projection-based filtering during HNSW construction
  - Our work adapts this idea specifically to NN-Descent's local-join structure
- **SHG** (Gong et al. 2025): shortcuts + compression for HNSW search
- STATUS: References exist from survey [24, 25, 26]. Need to re-read LSH-APG closely
  to correctly describe what they do vs. what we do.

### 2.3 NN-Descent Variants (~0.5 page)
- **PyNNDescent**: RP-tree init + optimized local join (Numba JIT)
- **EFANNA**: NN-Descent + KD-tree init + graph diversification
- **CAGRA** (NVIDIA): GPU-accelerated NN-Descent for graph construction
- Key point: NONE of these use distance filtering during local join. That's our gap.
- STATUS: Need to verify EFANNA and CAGRA details.

### What's READY
- [x] All references from survey paper
- [x] FINGER, LSH-APG, SHG descriptions (survey Section V)
- [x] NN-Descent algorithm understanding (from code)

### What's NEEDED
- [ ] Re-read LSH-APG paper to precisely describe their projection filter
- [ ] Verify CAGRA uses NN-Descent (cite NVIDIA paper)
- [ ] Verify no prior work applies filtering to NN-Descent specifically

---

## 3. Method (~1.5 pages)

### 3.1 Collision-Count Filter — Negative Result (~0.25 page)
- Idea: LSH fingerprints (128-bit) → Hamming distance as similarity proxy
- Filter rule: skip pair if collision_count < threshold
- Why it failed: discrete, noisy, poorly calibrated
- Show brief table: collision filter results vs. no filter
- Purpose in paper: motivates why we need a theoretically-grounded approach
- STATUS: Code exists (initializer.h CollisionTable).
  **NEED collision filter experimental results** — user will provide.

### 3.2 Projection-Based Distance Filter (~0.75 page)

**Math (from projection_filter.cpp):**

Given m random Gaussian vectors a_1, ..., a_m, define projections:
  P(x) = (a_1 · x, ..., a_m · x)

For two points u, v with true L2 distance d(u,v):
  ||P(u) - P(v)||^2 / d(u,v)^2  ~  chi^2(m) / m  (approximately)

By chi-squared concentration:
  P( ||P(u)-P(v)||^2 > t^2 * d(u,v)^2 ) < 1 - p_tau

where t^2 = chi^2_{p_tau}(m) (inverse CDF of chi-squared with m degrees of freedom).

**Filter condition:**
  Skip pair (u1, u2) if:
    proj_dist_sq(u1, u2) > t^2 * dk(u1)^2  AND  proj_dist_sq(u1, u2) > t^2 * dk(u2)^2

where dk(u) is the distance to u's current k-th nearest neighbor.

**Conservative AND logic:** Both points must agree the pair is far.

**Parameters:**
- m = number of projections (default: 32)
- p_tau = confidence level (default: 0.95)

- STATUS: Math is implemented and validated. Need to write up clearly with proper notation.

### 3.3 Modified NN-Descent Algorithm (~0.25 page)
- Pseudocode box showing where filter plugs into the local-join loop
- One-time setup: O(n * m * d) to compute all projections
- Per-pair filter cost: O(m) vs. O(d) for true distance
- When d >> m: huge savings. When d ≈ m: overhead ≈ savings.
- STATUS: Can extract pseudocode from nn_descent.cpp. Ready to write.

### 3.4 Complexity Analysis (~0.25 page)
- Memory: O(n * m) additional for projections
- Setup: O(n * m * d) one-time
- Per iteration: each candidate pair costs O(m) for filter check instead of O(d)
  - If filter rate = f, effective cost = (1-f)*O(d) + O(m) per pair
  - Net savings when f * d > m (approximately)
- STATUS: Ready to write from understanding.

### What's READY
- [x] Projection filter math (implemented + tested)
- [x] Algorithm pseudocode (from code)
- [x] Complexity analysis

### What's NEEDED
- [ ] Collision filter experimental results (to show it failed)
- [ ] Clean mathematical notation consistent with survey paper
- [ ] Cosine distance math: verify chi-squared bound applies or find correct bound
  - L2: well-established via JL lemma
  - Cosine: need to check — may require different concentration inequality
  - DECISION NEEDED: include cosine in this paper or leave for future work?

---

## 4. Experiments & Results (~2.5 pages)

### 4.1 Setup (~0.25 page)
- Datasets: GIST 1M (960-d), GIST 100K (960-d), SIFT 1M (128-d)
- Ground truth: brute-force via FAISS GPU
- Metrics: Recall@10, distance computations, wall-clock time
- Hardware: [NEED to document: CPU model, RAM, compiler flags]
- Parameters: k=10, mc=40, m=32 projections, random init (unless stated otherwise)
- STATUS: Setup is clear. Need to document hardware.

### 4.2 Main Result: Projection Filter Effectiveness (~0.75 page)

**Table: GIST 1M filter sweep** — DONE
| Config     | Recall | Dist Comps | Time (s) | Speedup | Recall Loss |
|------------|--------|------------|----------|---------|-------------|
| No filter  | 0.5894 | 1,551.5M   | 1056.7   | 1.00x   | 0.00%       |
| pτ = 0.99  | 0.5879 | 1,425.5M   | 1058.1   | 1.00x   | 0.25%       |
| pτ = 0.95  | 0.5820 | 1,265.7M   | 962.1    | 1.10x   | 1.26%       |
| pτ = 0.90  | 0.5720 | 1,143.0M   | 885.8    | 1.19x   | 2.95%       |
| pτ = 0.85  | 0.5603 | 1,048.4M   | 829.0    | 1.27x   | 4.94%       |
| pτ = 0.80  | 0.5470 | 968.3M     | 778.5    | 1.36x   | 7.19%       |

**Table: GIST 100K filter sweep (5-run averaged)** — DONE

**Table: SIFT 1M filter sweep** — DONE (5-run averaged)

**Plots needed:**
- [x] Recall vs. Distance Computations (GIST 1M) — exists in plot_all.py
- [x] Recall vs. Wall-Clock Time (GIST 1M) — exists
- [x] Cross-dataset speedup bar chart — exists
- [x] Distance computation savings bar chart — exists
- STATUS: Results and plots EXIST. May need to regenerate with better styling for paper.

### 4.3 Filter vs. Sampling Reduction (~0.5 page)
- Key argument: you can reduce dist_comps by lowering mc (max_candidates) too,
  but filtering gives BETTER recall per dist_comp
- Show: at ~same dist_comps, filter achieves higher recall than reduced sampling

**Table: GIST 1M MC sweep (no filter)** — DONE
| mc  | Recall | Dist Comps | Time (s) |
|-----|--------|------------|----------|
| 25  | 0.5207 | 1,250.9M   | 871.8    |
| 30  | 0.5490 | 1,364.9M   | 942.0    |
| 35  | 0.5713 | 1,463.7M   | 1003.6   |
| 40  | 0.5894 | 1,551.5M   | 1056.7   |

- **Plot:** Overlay filter sweep + mc sweep on same recall-vs-distcomp axes
- STATUS: Data DONE. Plot exists but may need refinement.

### 4.4 Dimensionality Analysis (~0.5 page)
- GIST (960-d): filter O(32) << distance O(960) → significant speedup
- SIFT (128-d): filter O(32) ≈ distance O(128) → marginal speedup
- This is a FEATURE, not a bug — explains exactly when to use the filter
- Shows the approach is most valuable where it's needed most (high-dim)
- STATUS: Results DONE. Analysis is straightforward.

### 4.5 Generality Across Initializers (~0.25 page)
- Show filter works with random init, LSH init, RP-tree init
- Even a small table (3 rows) showing comparable speedup across init methods
- STATUS: **NEED TO RUN** — currently all results use random init only.
  Suggested: run on GIST 100K with LSH init + filter, RP-tree init + filter.

### 4.6 [OPTIONAL] Cosine Distance Results (~0.25 page)
- If math checks out: run filter on GIST 100K with cosine distance
- Would strengthen the "general enhancement" claim
- STATUS: **NEED math verification first**, then experiments.
  DECISION: include or defer to future work?

### 4.7 [OPTIONAL] Collision Filter Results (~0.25 page)
- Brief table showing collision filter didn't help (or hurt recall with minimal speedup)
- Motivates the projection approach
- STATUS: **User will provide results.** Could go in Method section instead.

### What's READY for Experiments
- [x] GIST 1M: filter sweep (single run)
- [x] GIST 100K: filter sweep (5-run averaged)
- [x] SIFT 1M: filter sweep (5-run averaged)
- [x] GIST 1M: MC sweep (sampling baseline)
- [x] GIST 100K: MC sweep
- [x] All plots via plot_all.py

### What's NEEDED for Experiments
- [ ] **Collision filter results** (user providing)
- [ ] **Different initializer + filter runs** (GIST 100K: LSH init, RP-tree init)
- [ ] Hardware specification for paper
- [ ] [OPTIONAL] Cosine distance experiments
- [ ] [OPTIONAL] PyNNDescent comparison (DECISION PENDING with professor)
- [ ] Regenerate plots with paper-quality styling (fonts, labels, etc.)

---

## 5. Discussion (~0.5 page)

### Key Points to Make

1. **When it helps:** High-dimensional data where d >> m. The filter check O(m) is cheap
   relative to true distance O(d). GIST (960-d) shows clear benefit.

2. **When it doesn't:** Low-dimensional data (SIFT 128-d). Filter overhead ≈ savings.
   Recommendation: don't use for d < ~200.

3. **Construction-time recall vs. query-time quality:** The recall loss is at graph construction.
   For downstream tasks (HNSW search on the graph, CAGRA queries), a graph with 0.58 vs 0.59
   construction recall may perform identically at query time. This needs investigation but is a
   strong practical argument.

4. **Applicability:** This is a DROP-IN enhancement. PyNNDescent, EFANNA, CAGRA can all add
   projection filtering to their local-join loop. No structural changes needed.

5. **Memory overhead:** O(n * m) floats for projections. For n=1M, m=32: ~128MB. Modest.

- STATUS: All points are clear from results. Ready to write.

---

## 6. Conclusion & Future Work (~0.5 page)

### Conclusion
- Projection-based distance filtering reduces distance computations in NN-Descent by 18-37%
- Wall-clock speedup of 1.1-1.4x on high-dimensional data
- Theoretically grounded (chi-squared concentration), tunable (p_tau), conservative (AND logic)
- Drop-in enhancement for any NN-Descent variant

### Future Work
- **Cosine / inner-product distance:** Extend filter to non-L2 metrics
- **GPU integration:** Apply to CAGRA's GPU NN-Descent (projection check is SIMD-friendly)
- **Adaptive p_tau:** Start conservative (0.99) early iterations, increase aggressiveness as graph stabilizes
- **Learned projections:** Replace random Gaussian with data-dependent projections (PCA, etc.)
- **Query-time impact study:** Measure how construction recall loss affects downstream search quality

- STATUS: Ready to write.

---

## Figures Inventory

| Figure | Description | Status |
|--------|-------------|--------|
| Fig 1 | "Waste ratio" — fraction of dist_comps that don't update graph per iteration | **NEED TO BUILD** |
| Fig 2 | GIST 1M: Recall vs Dist Comps (filter sweep + mc sweep overlay) | EXISTS (plot_all.py) |
| Fig 3 | GIST 1M: Recall vs Wall-Clock Time | EXISTS |
| Fig 4 | Cross-dataset speedup bar chart | EXISTS |
| Fig 5 | Cross-dataset recall loss bar chart | EXISTS |
| Fig 6 | Distance computation savings (%) bar chart | EXISTS |
| Fig 7 | [OPTIONAL] Per-iteration filter rate evolution | COULD BUILD from iteration stats |
| Fig 8 | Algorithm pseudocode box | **NEED TO CREATE** |

---

## Open Decisions (for Professor)

1. **PyNNDescent comparison**: Include wall-clock comparison, or frame purely as
   "algorithmic enhancement measured by dist_comps saved"?
   - Risk of wall-clock comparison: their Python+NumPy is more optimized than our C++ prototype
   - Safe option: show dist_comp reduction, argue any NN-Descent implementation benefits

2. **Cosine distance**: Include in this paper or leave for future work?
   - Math needs verification (chi-squared bound may not directly apply)
   - If it works: strengthens the paper significantly
   - If it needs different bound: could be a full separate contribution

3. **Paper framing**:
   - Option A: "Enhancement to NN-Descent" (narrower, cleaner)
   - Option B: "Enhancement to any local-join graph construction" (broader claim, needs more evidence)

4. **Collision filter**: Include as negative result in Method, or just mention briefly?

---

## Immediate Action Items (Priority Order)

| # | Task | Effort | Blocking? |
|---|------|--------|-----------|
| 1 | Share collision filter results | User action | Yes (for Method section) |
| 2 | Run filter with LSH + RP-tree init on GIST 100K | ~2 hours compute | Yes (for generality claim) |
| 3 | Build "waste ratio" plot from iteration stats | ~1 hour coding | Yes (for intro Figure 1) |
| 4 | Document hardware specs | 5 min | Yes (for experiments) |
| 5 | Verify cosine distance math | ~2 hours research | Blocks cosine decision |
| 6 | Re-read LSH-APG paper | ~1 hour | Blocks related work |
| 7 | Verify CAGRA uses NN-Descent + get citation | ~30 min | Blocks intro paragraph |
| 8 | Regenerate plots with paper-quality styling | ~2 hours | No (can do last) |
| 9 | Decide PyNNDescent framing with professor | Discussion | Blocks experiments scope |

---

## References We'll Need (beyond survey)

Already have from survey:
- [Dong et al. 2011] NN-Descent
- [Malkov & Yashunin 2020] HNSW
- [Zhang et al. 2023] LSH-APG
- [Chen et al. 2023] FINGER
- [Gong et al. 2025] SHG
- [Jegou et al. 2011] PQ / Faiss
- [ANN-Benchmarks]

Need to add:
- [ ] CAGRA paper (NVIDIA, likely Ootomo et al. 2023 or similar)
- [ ] EFANNA paper (Fu & Cai, 2016)
- [ ] PyNNDescent (McInnes, 2018 — UMAP paper or pynndescent docs)
- [ ] Johnson-Lindenstrauss lemma (for projection theory)
- [ ] Chi-squared concentration bounds reference
