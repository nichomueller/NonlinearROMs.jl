# NonlinearROMs.jl — GNO Benchmark & Accuracy Tracking (Temporary)

> Temporary tracking file updated across the refactoring commits.

## Current Results (Accuracy & Correctness Fixes)

### 1. Test Suite (`Pkg.test("NonlinearROMs")`)
- **Result:** `1205 Pass | 0 Broken | 1205 Total` (`4m25.9s`)

### 2. CPU Micro-Benchmarks (`50x50`, `N=2550`, `E=26690`, `K=10`, `PlainMP`)
| Benchmark | Median Time | Memory | Allocs |
| :--- | :---: | :---: | :---: |
| `ZScore` (`D=6`) | `449.684 μs` | `1.17 MiB` | `29` |
| `gather_data` (`PlainMP`) | `9.524 ms` | `40.29 MiB` | `38` |
| `_integration` | `5.466 ms` | `20.96 MiB` | `12` |
| `_kv` Full-Matrix (`16x16`) | `116.061 ms` | `276.94 MiB` | `15` |
| `_kv` Channel-Wise (`16`) | `2.056 ms` | `16.29 MiB` | `3` |
| `EnergyLoss(H1(),fspace)` | `1.338 ms` | `2.62 MiB` | `31` |
| Full GNO Chain (2 layers, `FeatureKernel`) | `151.342 ms` | `301.07 MiB` | `213` |
| Full GNO Chain (2 layers, `SpatialKernel`) | `119.291 ms` | `251.42 MiB` | `215` |

### 3. Reactant / XLA Training (`50x50`, `PlainMP`, 2 layers, 200 epochs, `K=10`)
| Metric | Value |
| :--- | :---: |
| First epoch (JIT + training) | `132.033 s` |
| Training after first epoch (199 ep) | `26.281 s` |
| Training loop total | `158.313 s` |
| Setup and other overhead | `1.798 s` |
| Total call time | `160.111 s` |

### 4. 2D Poisson Diagnostic (`25x25`, 30 train / 10 test)
| Config | E | Compile (s) | Train (s) | Train L2 | Test L2 | Int L2 | Bnd L2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1-hop r=2h DiffMP 300ep` | `6924` | `5.71` | `14.57` | `0.11563` | `0.11019` | `0.10235` | `0.13008` |
| `1-hop r=2h PlainMP 300ep` | `6924` | `3.55` | `15.10` | `0.11268` | `0.11001` | `0.10286` | `0.12833` |
| `1-hop r=2h PlainMP 1500ep` | `6924` | `1.82` | `78.29` | `0.05789` | `0.08790` | `0.08328` | `0.09996` |
| `2-hop r=3h PlainMP 1500ep` | `16744` | `4.77` | `546.03` | `0.02046` | `0.09158` | `0.08844` | `0.10002` |
| `1-hop r=2h PlainMP H1-loss` | `6924` | `5.31` | `77.02` | `0.09194` | `0.06790` | `0.06509` | `0.07538` |
| `1-hop r=2h AnchoredMP H1+wd` | `6924` | `9.73` | `77.24` | `0.09408` | `0.07214` | `0.06923` | `0.07988` |
| `2-hop r=3h AnchoredMP wd` | `16744` | `13.18` | `518.18` | `0.01704` | `0.06043` | `0.05355` | `0.07680` |

### 5. Non-Local Multi-Mode 2D Poisson (`15x15`, 40 train / 10 test)
| Config | E | Compile (s) | Train (s) | Train L2 | Test L2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `NL 1-hop r=2h PlainMP 1000ep` | `1992` | `2.00` | `17.03` | `0.29842` | `0.35355` |
| `NL 3-hop r=3h PlainMP 1000ep` | `4544` | `15.77` | `160.32` | `0.10819` | `0.17546` |
| `NL 3-hop r=3h AnchoredMP Spat` | `4544` | `14.31` | `156.72` | `0.09608` | `0.14465` |

### 6. Analysis of Results

- **Fixing `EnergyLoss(H1)` (accuracy and speed):** Updating `EnergyLoss` to compute the standard Gram norm $(1/K)\sum_k e_k^T X e_k$ instead of $\|X e\|_2^2 = e^T X^2 e$ brings the `1-hop PlainMP H1-loss` test error down from `18.10%` to `6.79%` (train: `21.52% -> 9.19%`). In these runs, it performs better than MSE on the same setup (`6.79%` vs. `8.79%`). CPU evaluation time and memory usage also drop roughly by half (`2.57 ms / 4.85 MiB -> 1.34 ms / 2.62 MiB`), mainly because the function now only needs one sparse matrix-vector product per call instead of two.
- **Boundary behavior and multi-hop results on 2D Poisson (`25x25`):**
  - On `1-hop PlainMP 1500ep`, normalizing by the maximum degree (`1 / max_deg`) instead of each node's local degree (`1 / in_degree`)—along with adding residual connections—reduces test error from `13.03%` to `8.79%`. This seems to help mostly near the boundaries, where error drops from `15.41%` to `10.00%`.
  - On `2-hop r=3h`, combining `AnchoredMP` (`[dst; src - dst; w]`), `1 / max_deg` normalization, residual connections, a wider kernel (`24`), and `weight_decay = 1e-4` lowers test error from `9.30%` to `6.04%` (`intL2 = 5.36%`, `bndL2 = 7.68%`, `trL2 = 1.70%`).
- **Results on the multi-mode problem (`15x15`):** On this test case, `3-hop PlainMP` test error moves from `18.54%` to `17.55%`. Switching to `AnchoredMP` with `SpatialKernelFormatter` lowers it further to `14.47%` (`trL2 = 9.61%`), likely because a coordinate-only kernel $\kappa(x,y)$ is sufficient when the underlying PDE has constant coefficients.
- **Current performance and memory bottlenecks:**
  - The 2-layer CPU forward pass (`FeatureKernel`: `151.34 ms / 301.07 MiB`; `SpatialKernel`: `119.29 ms / 251.42 MiB`) and multi-hop XLA training (`2-hop`: `518–546 s` for 1500 epochs; `3-hop`: `157–160 s` for 1000 epochs) are still relatively slow. Most of the overhead seems to come from repeated 3D `permutedims`, calling `gather_data` at every layer, re-evaluating `SpatialKernel` per sample, and embedding graph topology arrays directly as inline MLIR constants.
  - The longer 2-hop XLA training time (`465 s -> 546 s`) is expected, as `kernel_layers` automatically scaled from `(16,)` to `(24,)` to match `hidden_layers = (24, 24, 24)`.
  - `ZScore` still allocates `1.17 MiB` because it makes a full `Float64` copy of the array.