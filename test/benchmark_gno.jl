
using BenchmarkTools
using Random
using Graphs
using Lux
using Gridap
using Gridap.FESpaces
using NonlinearROMs

using NonlinearROMs: TData, ZScore, _integration, _kv, build_model

println("==========================================================")
println("   NonlinearROMs.jl - Kernel Operator Benchmarks")
println("==========================================================")

# Reproducibility
Random.seed!(42)

# ==========================================================
# 1. Graph and data setup
# ==========================================================

# Setup a 50x50 Cartesian mesh matching the Poisson benchmark
domain = (0, 1, 0, 1)
partition = (50, 50)

model = CartesianDiscreteModel(domain, partition)
reffe = ReferenceFE(lagrangian, Float64, 1)

V = TestFESpace(
    model,
    reffe;
    conformity=:H1,
    dirichlet_tags=[1, 3, 7]
)

mesh_size = 1 / 50
gspace = GraphInducedSpace(V; radius=2.0 * mesh_size)
meas = InducedMeasure(gspace)
g = get_graph(gspace)

N = length(get_coords(gspace))
E = ne(g)

K = 10       # Number of training samples
D_in = 6     # 2 spatial coordinates + 4 PDE features
D_hid = 16   # Hidden state channels

println("Mesh configuration: N = $N nodes | E = $E edges | K = $K samples\n")

fixed_data = rand(TData, D_in, N, K)
dummy_out = rand(TData, 1, N, K)

edge_msgs = rand(TData, D_hid, E, K)
k_full = rand(TData, D_hid * D_hid, E, K)

# ==========================================================
# 2. CPU micro-benchmarks
# ==========================================================

println("1. ZScore computation and in-place normalisation (D=$D_in, N=$N, K=$K):")

b_zscore = @benchmark ZScore(d; normalise=true) setup=(d = copy($fixed_data))
display(b_zscore)

println("\n2. Graph message gathering (gather_data):")

b_gather = @benchmark gather_data($fixed_data, $gspace)
display(b_gather)

println("\n3. Graph message integration (_integration = scatter_data + degree weighting):")

b_integ = @benchmark _integration($edge_msgs, $meas)
display(b_integ)

println("\n4a. Kernel-state product _kv (Full-Matrix $D_hid x $D_hid per edge):")

b_kv_full = @benchmark _kv($k_full, $edge_msgs)
display(b_kv_full)

println("\n4b. Kernel-state product _kv (Channel-Wise $D_hid per edge):")

b_kv_cw = @benchmark _kv($edge_msgs, $edge_msgs)
display(b_kv_cw)

# ==========================================================
# 3. CPU full-chain forward pass
# ==========================================================

gno_model = KernelNeuralModel(
    gspace;
    hidden_layers=(16, 16, 16),
    activation=tanh
)

chain = build_model(gno_model, (fixed_data, dummy_out))
ps, st = Lux.setup(Random.default_rng(), chain)

println("\n5. Full GNO Chain Forward Pass (2 KernelNeuralLayers, CPU):")

b_chain = @benchmark $chain($fixed_data, $ps, $st)
display(b_chain)

# ==========================================================
# 4. Reactant / XLA training benchmark
# ==========================================================

println("\n==========================================================")
println("6. GNO Training on Reactant / XLA")
println("==========================================================")

# Use the same architecture and data as the CPU benchmark.
#
# 200 epochs match the Poisson benchmark configuration.
# Set this to 5 for a shorter XLA smoke test.

XLA_EPOCHS = 200

xla_solver = NeuralSolver(
    gno_model;
    epochs=XLA_EPOCHS,
    batch_size=0,        # Full batch
    verbose=true,
    print_every=50
)

# The logger records:
#   t_start      -> beginning of the training loop
#   t_start_fast -> completion of the first epoch
#
# The first epoch includes XLA compilation and its actual training work.

xla_logger = NonlinearROMs.get_logger(xla_solver)

# Use the library's own training entry point.
# The tensors are already prepared, so this isolates the GNO training
# path from FEM assembly and InputSpace/OutputSpace preprocessing.
#
# stats=nothing is sufficient here because this benchmark measures
# training only; it does not use the NeuralOperator online interface.

function run_xla_training(solver, inputs, targets)
    t0 = time()

    trained_op = NonlinearROMs.train(
        solver,
        (inputs, targets),
        nothing
    )

    return trained_op, time() - t0
end

println("\nStarting XLA training ($XLA_EPOCHS epochs)...")

xla_op, xla_call_time = run_xla_training(
    xla_solver,
    fixed_data,
    dummy_out
)

xla_end_time = time()

# Extract timings from the training logger.
xla_first_epoch_time =
    xla_logger.t_start_fast - xla_logger.t_start

xla_after_first_epoch_time =
    xla_end_time - xla_logger.t_start_fast

xla_training_loop_time =
    xla_end_time - xla_logger.t_start

xla_setup_time = max(
    0.0,
    xla_call_time - xla_training_loop_time
)

println("\nXLA Timing Summary")
println("------------------------------")
println("Epochs:                         $XLA_EPOCHS")
println("Training samples:               $K")
println("First epoch (JIT + training):   $(round(xla_first_epoch_time, digits=3)) s")
println("Training after first epoch:     $(round(xla_after_first_epoch_time, digits=3)) s")
println("Training loop total:            $(round(xla_training_loop_time, digits=3)) s")
println("Setup and other overhead:       $(round(xla_setup_time, digits=3)) s")
println("Total call time:                $(round(xla_call_time, digits=3)) s")

# ==========================================================
# 5. Post-training sanity check
# ==========================================================

println("\n7. Post-training sanity check:")

# NeuralOperator stores its trained chain and parameters on CPU.
# This checks that the trained model can still produce valid outputs.

pred, _ = xla_op.chain(
    fixed_data,
    xla_op.parameters,
    xla_op.states
)

@assert size(pred) == size(dummy_out)
@assert all(isfinite, pred)

println("SUCCESS! Output shape: $(size(pred))")
println("All predictions are finite.")

println("\n==========================================================")
println("Benchmark completed.")
println("==========================================================")
