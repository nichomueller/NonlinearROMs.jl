# integrate over an array

function integrate_over(a::AbstractArray,space)  where {T,N}
  array = zeros(T,size(a,2:N)...)
  for chunk in eachslice(a,dims=1)
    array += integrate_over(chunk,space)
  end
  return array
end

function integrate_over(a::AbstractVector,space)
  msg = "The first dimension of the array must match the number of coordinates in the function space"
  @check size(a,1) == num_coords(space) msg
  _integrate_over(a,space)
end

function _integrate_over(a,space)
  @abstractmethod
end

struct InducedMeasure{A} 
  space::A
end

function integrate_over(a::AbstractArray,meas::InducedMeasure)
  integrate_over(a,meas.space)
end

# function spaces

function _integrate_over(a,f::SimpleFESpace)
  integrate(FEFunction(f.space,a),f)
end

function _integrate_over(a,f::MultiSpace)
  cache = array_cache(f.coord_to_lcoords)
  int = zeros(eltype(a))
  @inbounds for i in eachindex(f.spaces)
    lcoords = getindex!(cache,f.coord_to_lcoords,i)
    int += _integrate_over(a[lcoords],f.spaces[i])
  end
  return int
end

# graphs

num_coords(g::AbstractGraph) = nv(g)
num_coords(g::NystromSubgraphIterator) = num_coords(g.graph)

function _integrate_over(a,g::AbstractGraph)
  int = zeros(eltype(a))
  for i in vertices(g)
    neighs = neighbors(g,i)
    nneigh = length(neighs)
    for j in neighs
      int += a[j] / nneigh
    end
  end
  return int
end

function _integrate_over(a,g::NystromSubgraphIterator)
  int = zeros(eltype(a))
  for gi in g
    int += _integrate_over(a,gi)
  end
  return int
end