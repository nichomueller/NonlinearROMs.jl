struct CoordsSampler{A<:Union{AbstractVector,Colon}}
  strategy::A
end

CoordsSampler() = CoordsSampler(:)

function get_coords(s::CoordsSampler,f::FunctionSpace)
  x = get_coords(f)
  view(x,s.strategy)
end

struct CoordStrategy{A,B}
  pass_coords::Val{A}
  sampler::CoordsSampler{B}
end

const NoCoord{B} = CoordStrategy{False,B}

function CoordStrategy(;pass_coords=Val{true}(),sampler=CoordsSampler())
  CoordStrategy(pass_coords,sampler)
end

function get_coords(cs::CoordStrategy,f::FunctionSpace)
  get_coords(cs.sampler,f)
end

abstract type GraphStrategy end

struct NoGraph <: GraphStrategy end

struct DistanceGraph{A<:Real}
  radius::A
end

DistanceGraph(;radius=1.0) = DistanceGraph(radius)

function build_graph(cs::CoordStrategy,gs::DistanceGraph,f::FunctionSpace)
  x = get_coords(cs,f)
  build_graph(x;radius=gs.radius)
end

struct DataFormatter{A<:CoordStrategy,B<:GraphStrategy}
  coord_strategy::A
  graph_strategy::B
end

function DataFormatter(;coord_strategy=CoordStrategy(),graph_strategy=NoGraph())
  DataFormatter(coord_strategy,graph_strategy)
end

# bundles

struct Bundle{A<:Tuple}
  bundle::A
end

const FeaturesBundle{A<:Tuple{Vararg{CellField}}} = Bundle{A}

function get_features(a::FeaturesBundle,f::FunctionSpace)
  features = init_features(f,a)
  @inbounds for (j,a) in enumerate(a.bundle)
    af = interpolation_values(a,f)
    set_features!(features,af,j)
  end
  return features
end

function get_features(cs::CoordStrategy,args...)
  features = get_features(args...)
  c = ntuple(i->Colon(),Val{ndims(features)-1}())
  view(features,cs.sampler,c...)
end

function fetch_data(cs::CoordStrategy,gs::DistanceGraph,f::FunctionSpace,a::FeaturesBundle)
  x = get_coords(cs,gs,f)
  features = get_features(cs,a,f)
  graph = build_graph(cs,gs,f)
  (x,features,graph)
end

function fetch_data(cs::NoCoord,gs::DistanceGraph,f::FunctionSpace,a::FeaturesBundle)
  features = get_features(cs,a,f)
  graph = build_graph(cs,gs,f)
  (features,graph)
end

function fetch_data(cs::CoordStrategy,gs::NoGraph,f::FunctionSpace,a::FeaturesBundle)
  x = get_coords(cs,gs,f)
  features = get_features(cs,a,f)
  (x,features)
end

function fetch_data(cs::NoCoord,gs::NoGraph,f::FunctionSpace,a::FeaturesBundle)
  features = get_features(cs,a,f)
  (features,)
end

function fetch_data(d::DataFormatter,f::FunctionSpace,a::FeaturesBundle)
  fetch_data(d.coord_strategy,d.graph_strategy,f,a)
end

# for the training, we also pass the snapshots values
function fetch_data(d::DataFormatter,f::FunctionSpace,a::FeaturesBundle,sols::Bundle)
  input = fetch_data(d,f,a)
  target = get_features(d.coord_strategy,sols,f)
  (input,target)
end

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
  nx = length(get_coords(f))
  nb = length(a.bundle)
  (nx,nb)
end

function _features_size(uf::SingleFieldParamFESpace,f,a)
  nx = length(get_coords(f))
  nb = length(a.bundle)
  np = param_length(uf)
  (nx,nb,np)
end

# function init_features(f,a::FeaturesBundle)
#   field = testitem(get_data(first(a.bundle)))
#   T = eltype2(get_vector_type(f))
#   s = _features_size(field,f,a)
#   zeros(T,s...)
# end

# function _init_features(::Field,f,a)
#   nx = length(get_coords(f))
#   nb = length(a.bundle)
#   (nx,nb)
# end

# function _init_features(item::GenericField{<:ParamFunction},f,a)
#   nx = length(get_coords(f))
#   nb = length(a.bundle)
#   np = num_params(item.object)
#   (nx,nb,np)
# end

# function _init_features(item::GenericField{<:TransientParamFunction},f,a)
#   nx = length(get_coords(f))
#   nt = num_times(item.object)
#   nb = length(a.bundle)
#   np = num_params(item.object)
#   (nx,nt,nb,np)
# end

function set_features!(cache,interp,j)
  @view cache[:,j] .= get_interp_values(interp)
end

function set_features!(cache,interp::ConsecutiveParamVector,j)
  @view cache[:,j,:] .= get_interp_values(interp)
end

#TODO missing interface for transient param functions