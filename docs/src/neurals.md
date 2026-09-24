## Neural Operators

In addition to classical linear ROMs, GridapROMs supports non-linear surrogate modeling via Neural Operators (DeepONet and NOMAD).
The integration relies on `NeuralReduction` and `NeuralSolver` to manage the offline training phase using XLA-accelerated backends (Lux.jl and Reactant.jl), and provides support for Continual and Transfer Learning via model fine-tuning.

## Transient Neural Operators

The space-time setting extends the Neural Operator capabilities of [`RBSteady`](@ref) to transient problems. 
The `reduced_operator` function builds spatiotemporal tensors by combining physical coordinates and time grids, allowing DeepONet and NOMAD architectures to predict dynamic fields.