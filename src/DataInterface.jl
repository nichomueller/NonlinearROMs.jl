struct CoordsSampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

CoordsSampler() = CoordsSampler(:)

function sample_coords(s::CoordsSampler,f::DiscreteSpace)
  x = get_coords(f)
  view(x,s.strategy)
end

struct DataFormatter{A,B}
  pass_coords::Val{A}
  sampler::CoordsSampler{B}
end

function DataFormatter(;pass_coords=Val{true}(),sampler=CoordsSampler())
  DataFormatter(pass_coords,sampler)
end

function fetch_data(d::DataFormatter,f::DiscreteSpace,args...)
  _fetch_data(d,f,args...)
end

function fetch_data(d::DataFormatter,f::GraphInducedSpace,args...)
  mat = _fetch_data(d,f,args...)
  
end

function _fetch_data(d::DataFormatter,f::InducedMeasure,args...)
  x = sample_coords(d.sampler,f)
  matrix_of_coords(x)
end

# bundles

struct Bundle{A<:Tuple}
  bundle::A
end

const FeaturesBundle{A<:Tuple{Vararg{CellField}}} = Bundle{A}

function get_features(a::FeaturesBundle,f::DiscreteSpace)
  features = init_features(f,a)
  @inbounds for (j,a) in enumerate(a.bundle)
    af = interpolation_values(a,f)
    set_features!(features,af,j)
  end
  return features
end

function sample_features(cs::CoordStrategy,args...)
  features = get_features(args...)
  c = ntuple(i->Colon(),Val{ndims(features)-1}())
  view(features,cs.sampler,c...)
end

function fetch_data(cs::CoordStrategy,f::DiscreteSpace,a::FeaturesBundle)
  x = get_coords(cs,f)
  features = get_features(cs,a,f)
  (x,features,graph)
end

function fetch_data(cs::DataFormatter{true},f::DiscreteSpace,a::FeaturesBundle)
  get_features(cs,a,f)
end

function fetch_data(d::DataFormatter,f::DiscreteSpace,a::FeaturesBundle)
  fetch_data(d.coord_strategy,d.graph_strategy,f,a)
end

# for the training, we also pass the snapshots values
function fetch_data(d::DataFormatter,f::DiscreteSpace,a::FeaturesBundle,sols::Bundle)
  input = fetch_data(d,f,a)
  target = get_features(d.coord_strategy,sols,f)
  (input,target)
end

struct LayerData{A,B,C}
  formatter::DataFormatter{A,B}
  metadata::C
end

LayerData(formatter::DataFormatter) = LayerData(formatter,nothing)

function fetch_data(d::LayerData,f::DiscreteSpace)
  fetch_data(d.coord_strategy,d.graph_strategy,f,a)
end

const BundleLayerData{A,B} = LayerData{A,B,<:Bundle}

# """
#     gather_edge_features(v,g::AbstractGraph;extra=nothing) -> (...,ne(g)) array

# Assembles the per-edge kernel input `[v(sender);v(receiver)]` (optionally with a
# static per-vertex feature array `extra`, e.g. a PDE coefficient `a`, concatenated
# in as `[v(sender);v(receiver);extra(sender);extra(receiver)]`), in `edges(g)` order
# -- i.e. it builds the `x` that makes `κ(x,y,v(x),v(y))` (or, with `extra`,
# `κ(x,y,a(x),a(y),v(x),v(y))`) a genuine per-edge, both-endpoints-dependent
# evaluation, which `scatter`/`_integration` then aggregate back down to
# one value per vertex. `v`/`extra` are `(channels,n_vertices)`, matching the
# Lux/`NeuralModel` `(channels,batch)` convention.
# """
# function gather_edge_features(v::AbstractMatrix,g::AbstractGraph;extra=nothing)
#   feats = isnothing(extra) ? v : vcat(v,extra)
#   din = size(feats,1)
#   out = zeros(eltype(feats),2*din,ne(g))
#   for (k,e) in enumerate(edges(g))
#     @views out[1:din,k] .= feats[:,src(e)]
#     @views out[din+1:end,k] .= feats[:,dst(e)]
#   end
#   return out
# end

# """
#     prepare_kernel_input(v,space) -> whatever the kernel network expects

# Converts the raw hidden state `v` (`(channels,n_points)`, values at *every* point
# of the function/graph `v` lives on) into whatever `num_coords(space)`-indexed input
# a `KernelLayer` needs to be evaluated on, dispatching on `space`:
# - FE-quadrature (`FunctionSpace`): identity -- `v` already is the nodal values at
#   that space's own coordinates, so no assembly is needed.
# - Graph: `gather_edge_features(v,space)` -- `κ(x,y,...)` depending on both
#   endpoints needs the per-edge `[v(sender);v(receiver)]` pair, not `v` itself.

# This is what lets `KernelNeuralLayer` call every `AbstractIntegralKernel` uniformly
# with the same raw `v`, regardless of which non-local discretization it uses.
# """
# prepare_kernel_input(v,f::FunctionSpace) = v
# prepare_kernel_input(v,g::AbstractGraph) = gather_edge_features(v,g)
# prepare_kernel_input(v,meas::InducedMeasure) = prepare_kernel_input(v,get_space(meas))

# utils

_underlying_space(f::SimpleSpace) = f.space
_underlying_space(f::MultiSpace) = _underlying_space(first(f.space))

function init_features(f,a)
  T = eltype2(get_vector_type(f))
  s = _features_size(_underlying_space(f),f,a)
  zeros(T,s...)
end

function _features_size(uf,f,a)
  @abstractmethod
end

function _features_size(uf::FESpace,f,a)
  nx = num_coords(f)
  nb = length(a.bundle)
  (nx,nb)
end

function _features_size(uf::SingleFieldParamFESpace,f,a)
  nx = num_coords(f)
  nb = length(a.bundle)
  np = param_length(uf)
  (nx,nb,np)
end

function set_features!(cache,interp,j)
  @view cache[:,j] .= get_interp_values(interp)
end

function set_features!(cache,interp::ConsecutiveParamVector,j)
  @view cache[:,j,:] .= get_interp_values(interp)
end

#TODO missing interface for transient param functions