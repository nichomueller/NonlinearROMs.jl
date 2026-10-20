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
InducedMeasure(space::MultiSpace{CommonDomain}) = InducedMeasure(GraphInducedSpace(space))

get_space(meas::FSpaceMeasure) = meas.space

function _integration(a::AbstractMatrix,meas::FSpaceMeasure{<:SimpleFESpace})
  space = get_space(meas)
  dof_values = to_dof_values(a,space)
  ah = FEFunction(space,dof_values)
  dΩ = get_measure(space)
  dc = integrate(ah,dΩ)
  assem = SparseMatrixAssembler(space,space)
  assemble_vector(assem,collect_cell_vector(dc))
end

function _integration(a::AbstractArray{T,3} where T,meas::FSpaceMeasure{<:SimpleFESpace})
  int = zeros(eltype(a),size(a,1),size(a,2))
  @inbounds for v in eachslice(a,dims=3)
    int .+= _integration(v,meas)
  end
  return int
end

function _integration(a::AbstractVector,meas::FSpaceMeasure{<:MultiSpace})
  space = get_space(meas)
  lcache = array_cache(space.coord_to_lcoords)
  int = zeros(eltype(a))
  @inbounds @views for i in eachindex(space.spaces)
    lcoords = getindex!(lcache,space.coord_to_lcoords,i)
    weighted = a[lcoords] ./ space.multiplicity[lcoords]
    int += _integration(weighted,space.spaces[i])
  end
  return int
end

function _integration(a::AbstractMatrix,meas::FSpaceMeasure{<:MultiSpace})
  space = get_space(meas)
  ccache = array_cache(space.lcoord_to_coords)
  lcache = array_cache(space.coord_to_lcoords)
  int = zeros(eltype(a),size(a,1))
  @inbounds @views for i in eachindex(space.spaces)
    ccoords = getindex!(ccache,space.lcoord_to_coords,i)
    lcoords = getindex!(lcache,space.coord_to_lcoords,i)
    weighted = a[:,lcoords] ./ space.multiplicity[lcoords]
    int[ccoords] += _integration(weighted,space.spaces[i])
  end
  return int
end

function get_measure(V::SingleFieldFESpace)
  trian = get_triangulation(V)
  degree = 2*get_polynomial_order(V)
  Measure(trian,degree)
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