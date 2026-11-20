# NonlinearROMs.jl — GNO Benchmark & Accuracy Tracking (Temporary)

> Temporary tracking file updated across the refactoring commits.

## Stage: Commit 1 — Baseline (Pre-Optimization, Extended Tests & Diagnostics)

### 1. Test Suite (`Pkg.test("NonlinearROMs")`)
- **Result:** `1155 Pass | 9 Broken | 1164 Total` (`4m20.8s`)

### 2. CPU Micro-Benchmarks (`50x50`, `N=2550`, `E=26690`, `K=10`, `PlainMP`)
| Benchmark | Median Time | Memory | Allocs |
| :--- | :---: | :---: | :---: |
| `ZScore` (`D=6`) | `449.076 μs` | `1.17 MiB` | `29` |
| `gather_data` (`PlainMP`) | `9.834 ms` | `40.29 MiB` | `38` |
| `_integration` | `5.636 ms` | `20.96 MiB` | `12` |
| `_kv` Full-Matrix (`16x16`) | `120.634 ms` | `276.94 MiB` | `15` |
| `_kv` Channel-Wise (`16`) | `2.463 ms` | `16.29 MiB` | `3` |
| `EnergyLoss(H1(),fspace)` | `2.566 ms` | `4.85 MiB` | `50` |
| Full GNO Chain (2 layers, `FeatureKernel`) | `116.514 ms` | `297.96 MiB` | `207` |
| Full GNO Chain (2 layers, `SpatialKernel`) | `94.047 ms` | `248.31 MiB` | `209` |

### 3. Reactant / XLA Training (`50x50`, `PlainMP`, 2 layers, 200 epochs, `K=10`)
| Metric | Value |
| :--- | :---: |
| First epoch (JIT + training) | `135.155 s` |
| Training after first epoch (199 ep) | `25.318 s` |
| Training loop total | `160.472 s` |
| Setup and other overhead | `1.765 s` |
| Total call time | `162.238 s` |

### 4. 2D Poisson Diagnostic (`25x25`, 30 train / 10 test)
| Config | E | Compile (s) | Train (s) | Train L2 | Test L2 | Int L2 | Bnd L2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1-hop r=2h DiffMP 300ep` | `6924` | `5.66` | `14.78` | `0.10804` | `0.09535` | `0.08992` | `0.10941` |
| `1-hop r=2h PlainMP 300ep` | `6924` | `3.64` | `15.10` | `0.10088` | `0.08990` | `0.08355` | `0.10603` |
| `1-hop r=2h PlainMP 1500ep` | `6924` | `1.88` | `75.59` | `0.05249` | `0.13034` | `0.12094` | `0.15413` |
| `2-hop r=3h PlainMP 1500ep` | `16744` | `4.48` | `465.42` | `0.02021` | `0.09296` | `0.08998` | `0.10100` |
| `1-hop r=2h PlainMP H1-loss` | `6924` | `4.96` | `76.45` | `0.21519` | `0.18104` | `0.17554` | `0.19593` |

### 5. Non-Local Multi-Mode 2D Poisson (`15x15`, 40 train / 10 test)
| Config | E | Compile (s) | Train (s) | Train L2 | Test L2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `NL 1-hop r=2h PlainMP 1000ep` | `1992` | `2.07` | `17.72` | `0.28987` | `0.35721` |
| `NL 3-hop r=3h PlainMP 1000ep` | `4544` | `16.53` | `144.89` | `0.10785` | `0.18536` |

### 6. Analysis of Results

- **`EnergyLoss(H1)` behavior:** On the 2D Poisson benchmark, training with `EnergyLoss(H1)` brings the reported loss value down to `5.33e-5`, while the actual relative $L^2$ error increases to `21.52%` (train) and `18.10%` (test)—roughly double the error seen with MSE on the same architecture. A likely explanation is that the current implementation evaluates $\|X e\|_2^2 = e^T X^2 e$ instead of the standard Gram norm $e^T X e$. Squaring the $H^1$ operator increases its condition number ($\kappa \sim h^{-4}$), which makes the optimization gradients much harder to follow.
- **Training length and overfitting on 2D Poisson (`25x25`):** At 300 full-batch epochs (`lr_max = 1e-3`), both `DifferenceMP` and `PlainMP` appear underfitted (`trL2 ≈ 10.1%–10.8%`, `teL2 ≈ 9.0%–9.5%`). Extending training to 1500 epochs (`lr_max = 5e-3`) shows that the model can fit the training data quite well (`trL2` drops to `5.25%` at 1-hop and `2.02%` at 2-hop), but it starts to overfit on this small dataset of 30 samples without regularization (`teL2` rises to `13.03%` at 1-hop and levels off around `9.30%` at 2-hop).
- **Higher errors near the boundaries:** Across all 2D Poisson runs, the boundary error (`Bnd L2`) is consistently `12%–27%` higher than the interior error (`Int L2`)—for example, `0.10603` vs. `0.08355` at 300 epochs, and `0.15413` vs. `0.12094` at 1500 epochs. This is probably due to using uniform `1 / in_degree` averaging: since boundary and corner nodes have fewer neighbors, each incoming edge gets a larger weight than its actual geometric area would warrant.
- **Effect of receptive field on the multi-mode problem (`15x15`):** Increasing the neighborhood from 1-hop (`r=2h`) to 3-hop (`r=3h`) cuts both training (`28.99% -> 10.79%`) and test (`35.72% -> 18.54%`) relative $L^2$ errors roughly in half. This suggests that wider message-passing is needed on problems where local information alone is not enough to infer the solution.
- **CPU memory and XLA runtime bottlenecks:**
  - A single 2-layer CPU forward pass (`FeatureKernel`) takes `116.51 ms` and allocates `297.96 MiB` (`207` allocations), largely due to repeated 3D array permutations and temporary edge-level arrays (`gather_data`: `9.83 ms / 40.29 MiB`; `_integration`: `5.64 ms / 20.96 MiB`).
  - `SpatialKernel` (`94.05 ms / 248.31 MiB`) is only slightly faster and lighter than `FeatureKernel` because the coordinate-only kernel is currently re-evaluated separately for each of the `K = 10` samples instead of being shared across the batch.
  - `ZScore` allocates `1.17 MiB` because it creates a full `Float64` copy of the input array.
  - On Reactant/XLA, the long `135.16 s` first epoch in Section 3 is mostly a one-time cold JIT startup cost; subsequent models compile in `1.88–16.53 s`. For multi-hop graphs, total runtime is mainly driven by the per-epoch step time (`465.42 s` for 1500 epochs on `2-hop, E=16744`, or ~`0.31 s/epoch`; `144.89 s` for 1000 epochs on `3-hop, E=4544`).