include("WeightedSimpleDiGraphs.jl")

abstract type GraphStrategy end

build_graph(s::GraphStrategy,V::FESpace) = @abstractmethod

struct MeshGraph <: GraphStrategy end

function build_graph(s::MeshGraph,V::FESpace)
  cell_to_dofs = Table(get_cell_dof_ids(V))
  dof_to_coords = get_free_dof_coordinates(V)
  g = WeightedSimpleDiGraph(length(dof_to_coords))
  build_graph!(g,s,cell_to_dofs,dof_to_coords)
end

function build_graph!(
  g::WeightedSimpleDiGraph,
  s::MeshGraph,
  cell_to_dofs::Table,
  dof_to_coords::AbstractVector{<:Point}
  )

  dof_to_cells = inverse_table(cell_to_dofs)
  cc = array_cache(cell_to_dofs)
  cd = array_cache(dof_to_cells)
  for dof in eachindex(dof_to_cells)
    cells = getindex!(cd,dof_to_cells,dof)
    for cell in cells
      dofs = getindex!(cc,cell_to_dofs,cell)
      for neighbor in dofs
        neighbor <= 0 && continue
        w = norm(dof_to_coords[dof] - dof_to_coords[neighbor])
        add_edge!(g,dof,neighbor,w)
      end
    end
  end
  g
end

struct DistanceGraph <: GraphStrategy
  radius::Real
end

DistanceGraph(;radius=1.0) = DistanceGraph(radius)

function build_graph(s::DistanceGraph,V::FESpace)
  dof_to_coords = get_free_dof_coordinates(V)
  data = map(x -> SVector(Tuple(x)),dof_to_coords)
  tree = BallTree(data)
  g = WeightedSimpleDiGraph(length(dof_to_coords))
  build_graph!(g,s,tree,dof_to_coords)
end

function build_graph(s::DistanceGraph,coords::AbstractMatrix{T}) where T
  points = [Point(Tuple(x)) for x in eachcol(coords)]
  build_graph(s,points)
end

function build_graph(s::DistanceGraph,coords::AbstractVector{<:Point})
  data = map(x -> SVector(Tuple(x)),coords)
  tree = BallTree(data)
  g = WeightedSimpleDiGraph{Int64,Float32}(length(coords))
  build_graph!(g,s,tree,coords)
end

function build_graph!(
  g::WeightedSimpleDiGraph,
  s::DistanceGraph,
  tree::NNTree,
  dof_to_coords::AbstractVector{<:Point}
  )

  for (dof,coord) in enumerate(dof_to_coords)
    dofs = inrange(tree,coord,s.radius)
    for neighbor in dofs
      w = norm(coord - dof_to_coords[neighbor])
      add_edge!(g,dof,neighbor,w)
    end
  end
  g
end

# Converts a graph into the format tensors `edge_index` and `edge_weights`required by message passing neural networks
function get_edge_tensors(graph)
  edge_index = Matrix{Int}(undef,2,ne(graph))
  edge_weights = Vector{Float32}(undef,ne(graph))
  
  k = 1
  for s in vertices(graph)
    for (d,w) in zip(outneighbors(graph,s),out_weights(graph,s))
      edge_index[1,k] = s
      edge_index[2,k] = d
      edge_weights[k] = w
      k += 1
    end
  end
  
  return edge_index,edge_weights
function sample_subgraph(g::AbstractGraph,l::Int)
  induced_subgraph(g,rand(1:nv(g),l))
end