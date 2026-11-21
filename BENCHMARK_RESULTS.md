# NonlinearROMs.jl — GNO Benchmark & Accuracy Tracking (Temporary)

> Temporary tracking file updated across the refactoring commits.

## Current Results (CPU & Reactant/XLA Optimizations)

### 1. Test Suite (`Pkg.test("NonlinearROMs")`)
- **Result:** `1213 Pass | 0 Broken | 1213 Total` (`4m39.3s`)

### 2. CPU Micro-Benchmarks (`50x50`, `N=2550`, `E=26690`, `K=10`, `PlainMP`)
| Benchmark | Median Time | Memory | Allocs |
| :--- | :---: | :---: | :---: |
| `ZScore` (`D=6`) | `244.422 μs` | `160 B` | `4` |
| `gather_data` (`PlainMP`) | `9.732 ms` | `40.29 MiB` | `38` |
| `_integration` | `5.565 ms` | `20.96 MiB` | `12` |
| `_kv` Full-Matrix (`16x16`) | `117.984 ms` | `276.94 MiB` | `15` |
| `_kv` Channel-Wise (`16`) | `2.255 ms` | `16.29 MiB` | `3` |
| `EnergyLoss(H1(),fspace)` | `1.333 ms` | `2.09 MiB` | `19` |
| Full GNO Chain (2 layers, `FeatureKernel`) | `83.563 ms` | `182.38 MiB` | `130` |
| Full GNO Chain (2 layers, `SpatialKernel`) | `37.010 ms` | `97.62 MiB` | `128` |

### 3. Reactant / XLA Training (`50x50`, `PlainMP`, 2 layers, 200 epochs, `K=10`)
> Note: this run was executed in an already warm Julia session. In a fresh session, the first epoch (cold JIT + training) takes ~`133 s`, with post-compilation training around `30–32 s`.

| Metric | Value |
| :--- | :---: |
| First epoch (warm JIT + training) | `3.348 s` |
| Training after first epoch (199 ep) | `31.951 s` |
| Training loop total | `35.299 s` |
| Setup and other overhead | `0.177 s` |
| Total call time | `35.476 s` |

### 4. 2D Poisson Diagnostic (`25x25`, 30 train / 10 test)
| Config | E | Compile (s) | Train (s) | Train L2 | Test L2 | Int L2 | Bnd L2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1-hop r=2h DiffMP 300ep` | `6924` | `5.69` | `15.12` | `0.11563` | `0.11019` | `0.10235` | `0.13008` |
| `1-hop r=2h PlainMP 300ep` | `6924` | `3.21` | `16.10` | `0.11268` | `0.11001` | `0.10286` | `0.12833` |
| `1-hop r=2h PlainMP 1500ep` | `6924` | `1.48` | `80.90` | `0.05789` | `0.08790` | `0.08328` | `0.09996` |
| `2-hop r=3h PlainMP 1500ep` | `16744` | `2.15` | `532.87` | `0.02044` | `0.09161` | `0.08848` | `0.10003` |
| `1-hop r=2h PlainMP H1-loss` | `6924` | `4.68` | `79.88` | `0.09194` | `0.06790` | `0.06509` | `0.07538` |
| `1-hop r=2h AnchoredMP H1+wd` | `6924` | `9.24` | `81.07` | `0.09408` | `0.07214` | `0.06923` | `0.07988` |
| `2-hop r=3h AnchoredMP wd` | `16744` | `12.24` | `531.15` | `0.01696` | `0.06029` | `0.05340` | `0.07669` |

### 5. Non-Local Multi-Mode 2D Poisson (`15x15`, 40 train / 10 test)
| Config | E | Compile (s) | Train (s) | Train L2 | Test L2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `NL 1-hop r=2h PlainMP 1000ep` | `1992` | `1.80` | `17.32` | `0.29840` | `0.35354` |
| `NL 3-hop r=3h PlainMP 1000ep` | `4544` | `16.33` | `167.22` | `0.10810` | `0.17539` |
| `NL 3-hop r=3h AnchoredMP Spat` | `4544` | `12.90` | `80.01` | `0.09593` | `0.14448` |

### 6. Observations & Limitations

- **CPU forward pass and memory:**
  - Replacing the full `Float64` copy in `ZScore` with a two-pass loop reduces median time from `~450 μs` to `~244 μs` and memory from `1.17 MiB` (`29` allocs) to `160 B` (`4` allocs).
  - Keeping edge tensors in `(D, K, E)` layout inside `IntegralKernel` and reusing `gather_data` across layers (`EdgeDataCache`) lowers the 2-layer CPU forward pass for `FeatureKernel` from `~151 ms / 301 MiB` to `~84 ms / 182 MiB` (`213 -> 130` allocs).
  - Evaluating `SpatialKernelFormatter` on `K = 1` and broadcasting in `_kv` lowers the 2-layer CPU forward pass from `~119 ms / 251 MiB` to `~37 ms / 98 MiB`.
  - Precomputing sparse triplets `(rows, cols, vals)` in `EnergyLoss` trims memory from `2.62 MiB` (`31` allocs) to `2.09 MiB` (`19` allocs) at roughly the same median runtime (`~1.33 ms`).
- **Reactant / XLA training:**
  - The clearest XLA speedup is on the non-local 3-hop `SpatialKernel` case (`15x15`, `K = 40`), where training time drops from `~157 s` to `~80 s` thanks to the `K = 1` kernel evaluation.
  - For feature-dependent kernels (`FeatureKernel`), per-epoch XLA training time did **not** improve meaningfully: `2-hop` training stays around `531–533 s`, and the 50x50 200-epoch test is slightly slower (`~30–32 s` vs. `~26 s` before). This suggests XLA was already fusing most permutations on its own, and that per-epoch cost on CPU is mostly limited by memory bandwidth over the large `(D, K, E)` edge-sample tensors.
  - Warm compilation times in Sections 4–5 are modestly lower in most cases (e.g., `2-hop PlainMP`: `4.77 s -> 2.15 s`), though the effect varies across configurations. Cold first-epoch compilation in a fresh Julia session remains around `~133 s`.
- **Consistency and caveats:**
  - All `1213` tests pass, and the accuracy metrics in Sections 4–5 match the previous run to 4 decimal places.
  - All timings are measured on a single CPU machine and show normal run-to-run variance.