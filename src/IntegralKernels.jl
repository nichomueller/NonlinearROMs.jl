abstract type AbstractIntegralKernel <: AbstractLuxLayer end

get_in_dimension(k::AbstractLuxLayer) = @abstractmethod
get_out_dimension(k::AbstractLuxLayer) = @abstractmethod

get_in_dimension(k::Lux.Dense) = k.in
get_out_dimension(k::Lux.Dense) = k.out
get_in_dimension(k::Lux.Conv) = k.in
get_out_dimension(k::Lux.Conv) = k.out

"""
    kv(k,v,meas::InducedMeasure)

Explicit `κ_θ(x,y,...)·v(y)` product in Kovachki et al.'s
`(K_t v)(x) = ∫ κ_θ(x,y,...) v(y) dy`: `k` is κ_θ's output, a flattened
`dout*din`-vector per evaluation point (edge, for a graph measure), and `v` is
the running hidden state at the SAME points `k` was evaluated at (gathered to
match `k`'s per-edge/per-point indexing by the `GraphMeasure` method below,
mirroring how `kernel_input` gathers data for feeding `κ_θ` itself). This batched
matrix-vector product is independent of whether `v` is ALSO fed as one of
`κ_θ`'s own inputs -- that's `formatter`'s/`pass_new_data`'s concern, orthogonal
to this one.
"""
function kv(k,v,meas)
  v′ = permutedims(v,(1,3,2))
  v′_src = NNlib.gather(v′,get_src_ids(meas))
  v_src = permutedims(v′_src,(1,3,2))
  _kv(k,v_src)
end

"""
    struct FourierKernel <: AbstractIntegralKernel
      in_dims::Int
      out_dims::Int
      truncation::Int
    end

FNO-style spectral convolution (Li et al.): `v` of size `(in_dims,N,K)` is
Fourier-transformed along the point axis, its first `truncation` modes are mixed
across channels by one real `(out_dims,in_dims)` matrix per mode (the
remaining modes are zeroed), and the result is transformed back to `N` points.
Real per-mode weights scale the real and imaginary parts of each mode alike:
in physical space they give an even (symmetric) convolution kernel.

The FFT treats the `N` points as uniformly spaced samples in the order they are
stored, so the kernel is only a convolution in physical space if the points are
ordered on a uniform 1D grid; for any other point set it is a (still trainable)
global operator whose meaning depends on the point ordering.
"""
struct FourierKernel <: AbstractIntegralKernel
  in_dims::Int
  out_dims::Int
  truncation::Int
end

get_in_dimension(l::FourierKernel) = l.in_dims
get_out_dimension(l::FourierKernel) = l.out_dims

function Lux.initialparameters(rng::AbstractRNG,l::FourierKernel)
  scale = TData(l.in_dims*l.out_dims)
  return (
    weight_re = rand(rng,TData,l.out_dims,l.in_dims,l.truncation) ./ scale,
    weight_im = rand(rng,TData,l.out_dims,l.in_dims,l.truncation) ./ scale,
  )
end

Lux.initialstates(rng::AbstractRNG,l::FourierKernel) = NamedTuple()

function (l::FourierKernel)((fixed_data,v),ps,st)
  N = size(v,2)
  v̂ = rfft(v,2) # (in_dims,N÷2+1,K)
  ŷ = _fft_modes(v̂,ps,l.truncation)
  y = irfft(ŷ,N,2)
  return (fixed_data,y),st
end

# Carries hoisted per-edge inputs across layers sharing the same formatter/measure
struct EdgeDataCache{F<:DataFormatter,A,B}
  fixed_data::A
  edge_data::B
  measure_id::UInt
  coord_dims::Int
  shared_coords::Bool
end

_unwrap_fixed_data(x) = x
_unwrap_fixed_data(x::EdgeDataCache) = x.fixed_data

"""
    struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure,C<:DataFormatter} <: AbstractIntegralKernel
      kernel::A
      measure::B
      formatter::C
      coord_dims::Int
    end

Wraps a plain network `kernel` into a non-local integral operator
`(K v)(x) = ∫ κ_θ(x,y,...) v(y) dy`, discretised by `measure`. Called as
`l((fixed_data,v),ps,st)`: `fixed_data` is the per-sample-fixed `(x,a(x))` tensor
assembled once upstream (see [`DataInterface.fetch_data`](@ref)), `v` is the
running hidden state from the previous layer. `formatter` (its own, layer-local
[`DataFormatter`](@ref)) decides which of `(x,a(x))`/`v` actually get concatenated
and fed to `kernel` -- different layers legitimately want different subsets
(Kovachki et al.'s three kernel cases `κ(x,y)`/`κ(x,y,a(x),a(y))`/`κ(x,y,v(x),v(y))`
are exactly a choice of `formatter`). `coord_dims` is how many of `fixed_data`'s
leading rows are coordinates (vs. features), which `formatter`'s
`pass_coords`/`pass_features` use to select which part of `fixed_data` to keep
(see [`DataInterface.merge_data`](@ref)). For coordinate-only formatters
(`pass_coords=Val(true)`, `pass_features=Val(false)`, `pass_new_data=Val(false)`,
e.g. [`SpatialKernelFormatter`](@ref)) with `shared_coords=true` and
`coord_dims > 0`, the spatial coordinates are assumed to be shared across
samples in a batch so `kernel` is evaluated on a single slice (`K=1`) and
broadcast in `_kv`. `fixed_data` passes through unchanged so the next layer can
use it too.
"""
struct IntegralKernel{A<:AbstractLuxLayer,B<:InducedMeasure,C<:DataFormatter} <: AbstractIntegralKernel
  kernel::A
  measure::B
  formatter::C
  coord_dims::Int
end

IntegralKernel(kernel,measure,formatter) = IntegralKernel(kernel,measure,formatter,0)

get_in_dimension(k::IntegralKernel) = get_in_dimension(k.kernel)
get_out_dimension(k::IntegralKernel) = get_out_dimension(k.kernel)

function Lux.initialparameters(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialparameters(rng,l.kernel)
end

function Lux.initialstates(rng::AbstractRNG,l::IntegralKernel)
  Lux.initialstates(rng,l.kernel)
end

function (l::IntegralKernel)((fixed_data,v),ps,st)
  mst = measure_state(l.measure)
  if isnothing(mst)
    fd = _unwrap_fixed_data(fixed_data)
    x = merge_data(l.formatter,fd,v,l.coord_dims)
    xk = gather_data(x,l.measure)
    k,st = evaluate(l.kernel,xk,ps,st)
    y = kv(k,v,l.measure)
    z = integration(y,l.measure)
    return (fd,z),st
  end
  (next_carrier,z),st = _apply_kernel(l,(fixed_data,v),ps,st,mst)
  return (_unwrap_fixed_data(next_carrier),z),st
end

_kernel_measure_state(::AbstractIntegralKernel) = nothing
_kernel_measure_state(k::IntegralKernel) = measure_state(k.measure)

function _apply_kernel(k::AbstractIntegralKernel,(carrier,v),ps,st,mst)
  k((_unwrap_fixed_data(carrier),v),ps,st)
end

function _apply_kernel(
  l::IntegralKernel{<:Any,<:Union{GraphMeasure,FESpaceMeasure,MultipleMeasure},F},
  (carrier,v),ps,st,mst
  ) where {F<:DataFormatter}

  fd = _unwrap_fixed_data(carrier)
  xk′,next_carrier = _get_edge_input_last(l,carrier,fd,v,mst)
  k′,st = evaluate(l.kernel,xk′,ps,st)
  v′ = permutedims(v,(1,3,2))
  v′_src = NNlib.gather(v′,mst.src_ids)
  y′ = _kv(k′,v′_src)
  z′ = _integration_last(y′,l.measure,mst)
  z = permutedims(z′,(1,3,2))
  return (next_carrier,z),st
end

function _get_edge_input_last(
  l::IntegralKernel{<:Any,<:Any,<:DataFormatter{<:Any,<:Any,true}},
  carrier,fd,v,mst
  )

  x = merge_data(l.formatter,fd,v,l.coord_dims)
  x′ = permutedims(x,(1,3,2))
  xk′ = _gather_data_last(x′,mst.src_ids,mst.dst_ids,mst.edge_weights,get_message_rule(l.measure))
  return xk′,carrier
end

function _get_edge_input_last(
  l::IntegralKernel{<:Any,<:Any,F},
  carrier,fd,v,mst
  ) where {A,B,F<:DataFormatter{A,B,false}}

  mid = objectid(l.measure)
  use_k1 = A && !B && l.formatter.shared_coords && l.coord_dims > 0 && size(fd,1) >= l.coord_dims
  if (
    carrier isa EdgeDataCache{F} &&
    carrier.measure_id == mid &&
    carrier.coord_dims == l.coord_dims &&
    carrier.shared_coords == use_k1 &&
    size(carrier.edge_data,3) == num_edges(l.measure)
  )
    return carrier.edge_data,carrier
  end
  x = merge_data(l.formatter,fd,v,l.coord_dims)
  # Opt-in K=1 fast path for coordinate-only kernels κ(x,y) on a shared mesh
  x′ = if use_k1
    permutedims(view(x,:,:,1:1),(1,3,2))
  else
    permutedims(x,(1,3,2))
  end
  xk′ = _gather_data_last(x′,mst.src_ids,mst.dst_ids,mst.edge_weights,get_message_rule(l.measure))
  return xk′,EdgeDataCache{F,typeof(fd),typeof(xk′)}(fd,xk′,mid,l.coord_dims,use_k1)
end

struct AttentionKernel{A<:AbstractLuxLayer,B<:AbstractLuxLayer,C<:InducedMeasure,D<:DataFormatter} <: AbstractIntegralKernel
  score::A
  value::B
  measure::C
  formatter::D
end

function Lux.initialparameters(rng::AbstractRNG,l::AttentionKernel)
  return (
    score = Lux.initialparameters(rng,l.score),
    value = Lux.initialparameters(rng,l.value),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::AttentionKernel)
  return (
    score = Lux.initialstates(rng,l.score),
    value = Lux.initialstates(rng,l.value),
  )
end

function (l::AttentionKernel)((fixed_data,v),ps,st)
  fd = _unwrap_fixed_data(fixed_data)
  x = merge_data(l.formatter,fd,v)
  xk = gather_data(x,l.measure)
  s,st_score = evaluate(l.score,xk,ps.score,st.score)
  val,st_value = evaluate(l.value,xk,ps.value,st.value)
  y = attention_integral(s,val,l.measure)
  return (fd,y),(score=st_score,value=st_value)
end

struct MultipleAttentionKernel{N,A<:NTuple{N,AbstractLuxLayer},B<:NTuple{N,AbstractLuxLayer},C<:AbstractLuxLayer,D<:InducedMeasure,E<:DataFormatter} <: AbstractIntegralKernel
  scores::A
  values::B
  aggregation::C
  measure::D
  formatter::E
end

function Lux.initialparameters(rng::AbstractRNG,l::MultipleAttentionKernel{N}) where N
  return (
    score = ntuple(i -> Lux.initialparameters(rng,l.scores[i]),Val(N)),
    value = ntuple(i -> Lux.initialparameters(rng,l.values[i]),Val(N)),
    aggregation = Lux.initialparameters(rng,l.aggregation),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::MultipleAttentionKernel{N}) where N
  return (
    score = ntuple(i -> Lux.initialstates(rng,l.scores[i]),Val(N)),
    value = ntuple(i -> Lux.initialstates(rng,l.values[i]),Val(N)),
    aggregation = Lux.initialstates(rng,l.aggregation),
  )
end

function (l::MultipleAttentionKernel{N})((fixed_data,v),ps,st) where N
  fd = _unwrap_fixed_data(fixed_data)
  x = merge_data(l.formatter,fd,v)
  xk = gather_data(x,l.measure)
  st_score = ()
  st_value = ()
  y = ()
  for i in 1:N
    si,st_scorei = evaluate(l.scores[i],xk,ps.score[i],st.score[i])
    vali,st_valuei = evaluate(l.values[i],xk,ps.value[i],st.value[i])
    st_score = (st_score...,st_scorei)
    st_value = (st_value...,st_valuei)
    y = (y...,attention_integral(si,vali,l.measure))
  end
  z,st_aggregation = evaluate(l.aggregation,cat(y...;dims=1),ps.aggregation,st.aggregation)
  return (fd,z),(score=st_score,value=st_value,aggregation=st_aggregation)
end

# implement the attention layer + RoPE

"""
    struct KernelNeuralLayer{A<:AbstractIntegralKernel,B<:Function} <: AbstractLuxLayer
      kernel::A
      in_dims::Int
      out_dims::Int
      activation::B
    end

One iterative layer of a kernel neural operator (Kovachki et al.):
```
v_{t+1}(x) = σ(W v_t(x) + (K_t v_t)(x) + b(x))
```
`kernel` (`K_t`) is the non-local integral operator --
any `AbstractIntegralKernel` (`IntegralKernel` wrapping a graph/FE-quadrature
`InducedMeasure`, `FourierKernel`, `AttentionKernel`, ...). The pointwise local
linear term `W`, bias `b`, and nonlinearity `σ` are owned by this layer, not by
`kernel`, since they're the same shape/role regardless of which non-local
discretization `kernel` uses. Called as `l((carrier,v),ps,st)`: `v` feeds both
the local `W v` term and `kernel` directly; `carrier` holds the per-sample
`(x,a(x))` tensor (which may be wrapped in an internal `EdgeDataCache` in the
returned tuple so downstream layers sharing the same formatter and measure reuse
gathered edge inputs). Precomputed measure topology and weights are stored in
`st.measure` so they are transferred once to `XDEV`.

`in_dims`/`out_dims` are the running hidden state's channel counts (`v_t`'s and
`v_{t+1}`'s), sizing `W`/`b` -- **not** `kernel`'s own internal input width.
`kernel`'s network sees `merge_data`'d/gathered data (often `v`'s channels plus
`fixed_data`'s, doubled again by `gather_data` for a graph measure), a completely
different, independently-sized tensor; `get_in_dimension(l.kernel)` reports that
internal width, not `v`'s, so it must not be used to size `W`.
"""
struct KernelNeuralLayer{A<:AbstractIntegralKernel,B<:Function} <: AbstractLuxLayer
  kernel::A
  in_dims::Int
  out_dims::Int
  activation::B
end

function KernelNeuralLayer(kernel::AbstractIntegralKernel,in_dims::Int,out_dims::Int;activation=identity)
  KernelNeuralLayer(kernel,in_dims,out_dims,activation)
end

function Lux.initialparameters(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialparameters(rng,l.kernel),
    weights = Lux.glorot_uniform(rng,l.out_dims,l.in_dims),
    bias = zeros(TData,l.out_dims,1,1),
  )
end

function Lux.initialstates(rng::AbstractRNG,l::KernelNeuralLayer)
  return (
    kernel = Lux.initialstates(rng,l.kernel),
    measure = _kernel_measure_state(l.kernel),
  )
end

function (l::KernelNeuralLayer)((carrier,v),ps,st)
  (next_carrier,kout),kst = _apply_kernel(l.kernel,(carrier,v),ps.kernel,st.kernel,st.measure)
  sz = size(v)
  wout = reshape(ps.weights * reshape(v,sz[1],:),l.out_dims,Base.tail(sz)...)
  y = l.activation.(wout .+ kout .+ ps.bias)
  return (next_carrier,y),(kernel=kst,measure=st.measure)
end

# swapping the discretisation (e.g. to evaluate on a new mesh), keeping the
# trained network: callers must also refresh `st.measure` via `replace_measure_state`
# (see `replace_space`)

replace_measure(l::AbstractLuxLayer,meas::InducedMeasure) = l

function replace_measure(c::Lux.Chain,meas::InducedMeasure)
  Lux.Chain(map(l -> replace_measure(l,meas),c.layers))
end

function replace_measure(l::KernelNeuralLayer,meas::InducedMeasure)
  KernelNeuralLayer(replace_measure(l.kernel,meas),l.in_dims,l.out_dims,l.activation)
end

function replace_measure(k::IntegralKernel,meas::InducedMeasure)
  _check_measure(k.measure,meas)
  IntegralKernel(k.kernel,meas,k.formatter,k.coord_dims)
end

function replace_measure(k::AttentionKernel,meas::InducedMeasure)
  _check_measure(k.measure,meas)
  AttentionKernel(k.score,k.value,meas,k.formatter)
end

function replace_measure(k::MultipleAttentionKernel,meas::InducedMeasure)
  _check_measure(k.measure,meas)
  MultipleAttentionKernel(k.scores,k.values,k.aggregation,meas,k.formatter)
end

_get_message_rule(::AbstractLuxLayer) = nothing
_get_message_rule(l::KernelNeuralLayer) = _get_message_rule(l.kernel)
_get_message_rule(k::Union{IntegralKernel,AttentionKernel,MultipleAttentionKernel}) = get_message_rule(k.measure)

function _get_message_rule(c::Lux.Chain)
  for l in c.layers
    r = _get_message_rule(l)
    !isnothing(r) && return r
  end
  return nothing
end

replace_measure_state(::AbstractLuxLayer,st) = st

function replace_measure_state(c::Lux.Chain,st::NamedTuple)
  NamedTuple{keys(st)}(map(replace_measure_state,values(c.layers),values(st)))
end

function replace_measure_state(l::KernelNeuralLayer,st::NamedTuple)
  (kernel=st.kernel,measure=_kernel_measure_state(l.kernel))
end

function _check_measure(old::InducedMeasure,new::InducedMeasure)
  kold,knew = _space_kind(get_space(old)),_space_kind(get_space(new))
  msg = "The new measure must be of the same kind as the trained one: got $knew, trained on $kold"
  @check kold == knew msg
end

# a MultipleSpaces is of the kind of its local spaces, so that e.g. a model trained on
# several graphs can be evaluated on a single new one
_space_kind(f::DiscreteSpace) = nameof(typeof(f))
_space_kind(f::MultipleSpaces) = _space_kind(first(f.lspaces))

# utils

"""
    get_kernel_channels(kernel_formatter,fixed_formatter,ncoords,fixed_dims)

Rows of the fixed per-sample input (laid out as `[x;a(x)]` by `fixed_formatter`,
with `ncoords` coordinate channels and `fixed_dims` channels overall) that
`kernel_formatter`'s `pass_coords`/`pass_features` ask to feed to `κ_θ`. A part
`fixed_formatter` doesn't pass in the first place can't be selected, and is
silently skipped. Returns `:` when everything is selected, `nothing` when nothing is.
"""
function get_kernel_channels(
  kernel_formatter::DataFormatter{KC,KF},
  fixed_formatter::DataFormatter{FC},
  ncoords::Int,
  fixed_dims::Int
  ) where {KC,KF,FC}

  nc = FC ? ncoords : 0
  coords = KC ? (1:nc) : (1:0)
  features = KF ? ((nc+1):fixed_dims) : (1:0)
  if isempty(coords) && isempty(features)
    nothing
  elseif length(coords)+length(features) == fixed_dims
    Colon()
  elseif isempty(features)
    coords
  elseif isempty(coords)
    features
  else
    @check last(coords)+1 == first(features)
    first(coords):last(features)
  end
end

select_channels(data,::Colon) = data
select_channels(data,::Nothing) = nothing
select_channels(data::AbstractInputData,ids::AbstractUnitRange) = view(data,ids,:,:)

function attention_integral(scores,values,meas::InducedMeasure)
  @notimplemented "Attention integration is not implemented for measure type $(typeof(meas))."
end

function attention_integral(scores,values,meas::GraphMeasure)
  α = attention(vec(scores),meas)
  scatter_data(α'.*values,meas)
end

function attention_integral(scores,values,meas::MultipleMeasure)
  blockwise(attention_integral,meas,scores,values)
end

# reconstructs the (dout x din) matrix flattened into k's first axis and applies
# it to v, batched over the two trailing axes (either (E,K) or (K,E), allowing
# singleton broadcast axes in k such as K=1)
function _kv(k::AbstractInputData,v::AbstractInputData)
  din,n2,n3 = size(v)
  kdim = size(k,1)
  if kdim == din
    return k .* v
  else
    dout = div(kdim,din)
    k_r = reshape(k,dout,din,size(k,2),size(k,3))
    v_r = reshape(v,1,din,n2,n3)
    return dropdims(sum(k_r .* v_r,dims=2),dims=2)
  end
end

function _fft_modes(v̂,ps,k)
  nf = size(v̂,2)
  m = min(k,nf)
  v̂m = permutedims(v̂[:,1:m,:],(1,3,2)) # (in_dims,K,m)
  vr,vi = real.(v̂m),imag.(v̂m)
  Wr = ps.weight_re[:,:,1:m]
  Wi = ps.weight_im[:,:,1:m]
  # per mode: ŷ = W*v̂, with W = Wr + i*Wi
  yr = batched_mul(Wr,vr) .- batched_mul(Wi,vi)
  yi = batched_mul(Wr,vi) .+ batched_mul(Wi,vr)
  ŷ = permutedims(complex.(yr,yi),(1,3,2)) # (out_dims,m,K)
  m == nf ? ŷ : cat(ŷ,_zeros_like(ŷ,size(ŷ,1),nf-m,size(ŷ,3));dims=2)
end

function _zeros_like(x,dims...)
  z = similar(x,dims...)
  fill!(z,zero(eltype(z)))
  z
end
