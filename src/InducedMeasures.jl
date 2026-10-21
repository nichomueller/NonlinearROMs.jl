dimension(f::FunctionSpace) = num_coords(f)
dimension(g::AbstractGraph) = ne(g) # graphs are evaluated per-edge

abstract type InducedMeasure end

InducedMeasure(space::DiscreteSpace) = @abstractmethod

CellData.integrate(a,meas::InducedMeasure) = integration(a,meas)

integration(a,meas::InducedMeasure) = @abstractmethod

get_space(meas::InducedMeasure) = @abstractmethod

function integration(a::AbstractArray{T,3} where T,meas::InducedMeasure)
  msg = "Integrand must be of size (latent_dim, num_coords, num_params), where num_coords == $(num_coords(meas)); got size $(size(a))"
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

function _integration(a::AbstractArray{T,3} where T,meas::FSpaceMeasure{<:SimpleFESpace})
  space = get_space(meas)
  dΩ = get_measure(space)
  v = get_fe_basis(space)
  assem = SparseMatrixAssembler(space,space)
  b = similar(a)
  @inbounds @views for d in axes(a,1)
    aμ = ConsecutiveParamArray(a[d,:,:])
    bμ = ConsecutiveParamArray(b[d,:,:])
    ahμ = FEFunction(space,aμ)
    dc = integrate(ahμ*v,dΩ)
    assemble_vector!(bμ,assem,collect_cell_vector(space,dc))
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

get_space(meas::GraphMeasure) = meas.space
get_graph(meas::GraphMeasure) = get_graph(get_space(meas))

function _integration(a::AbstractArray,meas::GraphMeasure)
  space = get_space(meas)
  out = scatter_data(a,space)
  _normalise!(out,meas.weights)
  return out
end

# utils 

function _normalise!(a::AbstractVector,weights::AbstractVector)
  for i in eachindex(a)
    !iszero(weights[i]) && (a[i] /= weights[i])
  end
  a
end

function _normalise!(a::AbstractMatrix,weights::AbstractVector)
  @views for i in eachindex(a)
    !iszero(weights[i]) && (a[:,i] ./= weights[i])
  end
  a
end