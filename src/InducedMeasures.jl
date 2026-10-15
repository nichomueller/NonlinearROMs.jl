dimension(f::FunctionSpace) = num_coords(f)
dimension(g::AbstractGraph) = ne(g) # graphs are evaluated per-edge

abstract type InducedMeasure end

InducedMeasure(space::DiscreteSpace) = @abstractmethod

CellData.integrate(a,meas::InducedMeasure) = integration(a,meas)

integration(a,meas::InducedMeasure) = @abstractmethod

get_space(meas::InducedMeasure) = @abstractmethod

function integration(a::AbstractArray{T,N},meas::InducedMeasure) where {T,N}
  latent...,dim = size(a)
  integration(reshape(a,prod(latent),dim),meas)
end

function integration(a::AbstractVecOrMat,meas::InducedMeasure)
  msg = "The array's final dimension must match the number of coordinates in the function space"
  dim = size(a,ndims(a))
  @check dim == dimension(get_space(meas)) msg
  _integration(a,meas)
end

function _integration(a::AbstractVecOrMat,meas::InducedMeasure)
  @abstractmethod
end

struct FSpaceMeasure{A<:FunctionSpace} <: InducedMeasure
  space::A
end

InducedMeasure(space::FunctionSpace) = FSpaceMeasure(space)
InducedMeasure(space::MultiSpace{CommonDomain}) = InducedMeasure(GraphInducedSpace(space))

get_space(meas::FSpaceMeasure) = meas.space

function _integration(a::AbstractVector,meas::FSpaceMeasure{<:SimpleFESpace})
  space = get_space(meas)
  ah = FEFunction(space,a)
  integrate(ah,space)
end

function _integration(a::AbstractMatrix,meas::FSpaceMeasure{<:SimpleFESpace})
  int = zeros(eltype(a),size(a,1))
  @inbounds for col in eachcol(a)
    int .+= _integration(col,meas)
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

struct GraphMeasure{A<:GraphInducedSpace,B<:AbstractVector} <: InducedMeasure
  graph::A
  weights::B
end

function InducedMeasure(graph::GraphInducedSpace)
  w = zeros(Int,nv(graph))
  for e in edges(graph)
    w[dst(e)] += 1
  end
  return GraphMeasure(graph,w)
end

get_space(meas::GraphMeasure) = meas.graph

function _integration(a::AbstractArray,meas::GraphMeasure)
  graph = get_space(meas)
  out = scatter(a,graph)
  _normalise!(out,meas.weights)
  return out
end

# utils 

function _normalise!(a::AbstractVector,weights::AbstractVector)
  for i in eachindex(a)
    weights[i] > 0 && (a[i] /= weights[i])
  end
  a
end

function _normalise!(a::AbstractMatrix,weights::AbstractVector)
  @views for i in eachindex(a)
    weights[i] > 0 && (a[:,i] ./= weights[i])
  end
  a
end