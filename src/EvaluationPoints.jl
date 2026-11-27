const InterpolableFunction{M,A<:Function} = Interpolable{M,A}

function Arrays.evaluate!(cache,f::InterpolableFunction,x::Point)
  fx = evaluate!(cache,f.uh,x)
  return fx
end

function Arrays.return_cache(f::InterpolableFunction,x::AbstractVector{<:Point})
  fx = evaluate(f,testitem(x))
  cache = 
  return fx
end

abstract type EvaluationPoints{D} end

EvaluationPoints(args...) = @abstractmethod

discretise(data,points::EvaluationPoints) = @abstractmethod

struct NodalPoints{D,A<:AbstractVector{<:Point{D}}} <: EvaluationPoints{D}
  points::A
end

function EvaluationPoints(points::AbstractVector{<:Point})
  NodalPoints(points)
end

for f in (:EvaluationPoints,:NodalPoints)
  @eval begin
    function $f(x::AbstractArray{T,D}) where {T,D}
      msg = """
      Unsupported type of points. EvaluationPoints can only be constructed from a vector 
      of Point objects or an array of coordinates with isbits eltype. 
      """
      isbits(T) || @notimplemented msg
      points = map(Point,eachslice(x,dims=1))
      NodalPoints(points)
    end
  end
end

function discretise(data,points::NodalPoints)
  ah = Interpolable(data)
  ax = evaluate(ah,points.points)
  dataset(ax)
end

struct CellPoints{D,A<:AbstractVector{<:AbstractVector{<:Point{D}}}} <: EvaluationPoints{D}
  cellpoints::A
end

for f in (:EvaluationPoints,:CellPoints), T in (:Grid,:CellDatum,:Measure)
  @eval begin
    function $f(a::$T)
      x = get_cell_points(a)
      CellPoints(get_data(x))
    end
  end
end

struct CellDofs{D,A<:CellDof} <: EvaluationPoints{D}
  celldofs::A
  function CellDofs(celldofs::A) where A<:CellDof
    D = _get_dim(testitem(get_data(dofs)))
    new{D,A}(celldofs)
  end
end

function EvaluationPoints(V::FESpace)
  celldofs = get_fe_dof_basis(V)
  dof_basis = testitem(get_data(celldofs))
  isnodal = _is_nodal_dof(dof_basis)
  isconfo = _is_conforming(V)
  if isnodal && isconfo
    NodalPoints(get_points(V))
  elseif isconfo
    CellPoints(get_triangulation(V))
  else
    CellDofs(celldofs)
  end
end

# utils

_get_dim(x::Point{D}) where D = D
_get_dim(basis::AbstractArray{<:Dof}) = _get_dim(get_nodes(basis))

_is_nodal_dof(basis::AbstractArray{T}) where T = isa(T,PointValue)

function _is_conforming(V::FESpace)
  ids = get_cell_dof_ids(V)
  cache = array_cache(ids)
  seen = Set{Int}()
  for cell in eachindex(ids)
    cell_ids = getindex!(cache,ids,cell)
    for dof in cell_ids
      dof in seen && return true
      push!(seen,dof)
    end
  end
  return false
end