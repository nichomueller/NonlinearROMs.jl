dimension(f::FunctionSpace) = num_coords(f)
dimension(g::AbstractGraph) = ne(g) # graphs are evaluated per-edge
dimension(space::AbstractGraphInducedSpace) = dimension(get_graph(space))

abstract type InducedMeasure end

InducedMeasure(space::DiscreteSpace) = @abstractmethod

CellData.integrate(a,meas::InducedMeasure) = integration(a,meas)

integration(a,meas::InducedMeasure) = @abstractmethod

get_space(meas::InducedMeasure) = @abstractmethod

function integration(a::InputData,meas::InducedMeasure)
  msg = "Integrand must be of size (latent_dim, num_coords, num_params), where num_coords == $(dimension(get_space(meas))); got size $(size(a))"
  @check size(a,2) == dimension(get_space(meas)) msg
  _integration(a,meas)
end

function _integration(a::AbstractArray,meas::InducedMeasure)
  @abstractmethod
end

struct FSpaceMeasure{A<:FunctionSpace} <: InducedMeasure
  space::A
end

InducedMeasure(space::FunctionSpace) = FSpaceMeasure(space)

get_space(meas::FSpaceMeasure) = meas.space

function _integration(a::InputData,meas::FSpaceMeasure{<:FunctionSpace{<:FESpace}})
  space = get_space(meas)
  fespace = space.space
  dΩ = get_measure(space)
  v = get_fe_basis(fespace)
  assem = SparseMatrixAssembler(fespace,fespace)
  b = similar(a)
  @inbounds @views for d in axes(a,1)
    aμ = ConsecutiveParamArray(a[d,:,:])
    bμ = ConsecutiveParamArray(b[d,:,:])
    ahμ = FEFunction(fespace,aμ)
    dc = integrate(ahμ⋅v,dΩ)
    assemble_vector!(bμ,assem,collect_cell_vector(fespace,dc))
  end
  return b
end

struct GraphMeasure{A<:GraphInducedSpace,B<:AbstractVector} <: InducedMeasure
  space::A
  weights::B
end

function InducedMeasure(space::GraphInducedSpace)
  graph = get_graph(space)
  w = zeros(Int,nv(graph))
  for e in edges(graph)
    w[dst(e)] += 1
  end
  return GraphMeasure(space,w)
end

function InducedMeasure(space::MultiGraphInducedSpace)
  graph = get_graph(space)
  w = zeros(Int,nv(graph))
  cache = array_cache(space.node_to_lnode)
  for e in edges(graph)
    v = dst(e)
    lnodes = getindex!(cache,space.node_to_lnode,v)
    w[v] += length(lnodes)
  end
  return GraphMeasure(space,w)
end

get_space(meas::GraphMeasure) = meas.space
get_graph(meas::GraphMeasure) = get_graph(get_space(meas))

function _integration(a::InputData,meas::GraphMeasure)
  space = get_space(meas)
  out = scatter_data(a,space)
  _normalise!(out,meas.weights)
  return out
end

# utils 

function _normalise!(a::InputData,weights::AbstractVector)
  @views for i in axes(a,2)
    !iszero(weights[i]) && (a[:,i,:] ./= weights[i])
  end
  a
end