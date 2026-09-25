function collect_data(solver,op,r,args...)
  @abstractmethod
end

function collect_data(solver::NeuralSolver,op::ParamOperator,r::Realisation,args...)
  trial = get_trial(op)
  values = solve(solver,op,r,args...)
  coords = get_free_dof_coordinates(trial)
  build_dataset(solver,values,coords,r)
end

function build_dataset(solver::NeuralSolver,inputs...)
  sinputs = sample(solver,inputs...)
  fsinputs = get_inputs(Float32,sinputs...)
  _build_dataset(solver,fsinputs...)
end

function _build_dataset(::NeuralSolver{<:KernelReduction},values,coords,params)
  dataset = DataCollector(data,coords,params)
  return dataset
  
end

# utils

function get_inputs_and_stats(args...)
  inputs = get_inputs(args...)
  stats = Normalisation(inputs;normalise=true)
  return inputs,stats
end

function get_inputs(args...)
  get_inputs(Float32,args...)
end

function get_inputs(::Type{T},values,coords::AbstractVector{<:Point},r::Realisation) where T
  d = T.(matrix_of_values(values))
  x = T.(matrix_of_coords(coords))
  p = T.(matrix_of_params(r))
  return (d,x,p)
end

function get_inputs(::Type{T},values,r::Realisation) where T
  d = T.(matrix_of_values(values))
  p = T.(matrix_of_params(r))
  return (d,p)
end

function get_inputs(::Type{T},values,coords::AbstractVector{<:Point},r::TransientRealisation) where T
  times = get_times(r)
  d = get_formatted_data(T,values,times)
  x = T.(matrix_of_coords(coords,times))
  p = T.(matrix_of_params(r))
  return (d,x,p)
end

function get_inputs(::Type{T},values,r::TransientRealisation) where T
  times = get_times(r)
  d = get_formatted_data(T,values,times)
  p = T.(matrix_of_params(r))
  return (d,p)
end

function get_formatted_data(::Type{T},values,times) where T
  d = T.(matrix_of_values(values))
  nx = size(d,1)
  nt = length(times)
  np = Int(size(d,2)/nt)
  reshape(permutedims(reshape(d,nx,np,nt),(1,3,2)),:,np)
end

function get_inputs(::Type{T},coords::AbstractVector{<:Point},r::AbstractRealisation) where T
  x = T.(matrix_of_coords(coords))
  p = T.(matrix_of_params(r))
  return (x,p)
end

dimension(μ::Realisation) = length(first(μ))
dimension(μ::TransientRealisation) = dimension(get_params(μ))

function matrix_of_values(x::AbstractParamArray)
  get_all_data(x)
end

function matrix_of_params(r::AbstractRealisation)
  params = zeros(dimension(r),num_params(r))
  matrix_of_params!(params,r)
end

function matrix_of_params!(params,r::AbstractRealisation)
  @check size(params,2) == num_params(r)
  μ = get_params(r)
  @inbounds @views for i in axes(params,2)
    params[:,i] = μ.params[i]
  end
  params
end

function matrix_of_coords(coords::AbstractVector{Point{D,T}}) where {D,T}
  coords_mat = zeros(T,D,length(coords))
  for (i,coord) in enumerate(coords)
    for d in 1:D 
      coords_mat[d,i] = coord.data[d]
    end
  end
  return coords_mat
end

function matrix_of_coords(coords::AbstractVector{Point{D,T}},times::AbstractVector{S}) where {D,T,S}
  TS = promote_type(T,S)
  coords_mat = zeros(TS,D+1,length(coords)*length(times))
  col = 0
  for t in times,coord in coords
    col += 1
    for d in 1:D
      coords_mat[d,col] = coord.data[d]
    end
    coords_mat[D+1,col] = t
  end
  return coords_mat
end

#TODO @Isaia: your old tensor_of_coords function stacked params before the coords 
# on the rows, are you sure it's correct? I am doing the opposite here, please fix it 
# in case it's wrong.
function tensor_of_coords(coords::AbstractMatrix{T},params::AbstractMatrix{S}) where {T,S}
  TS = promote_type(T,S)
  D,nx = size(coords)
  P,np = size(params)
  tensor = zeros(TS,D+P,nx,np)
  for (i,x) in enumerate(eachcol(coords))
    for (j,μ) in enumerate(eachcol(params))
      for d in 1:D
        tensor[d,i,j] = x[d]
      end
      for p in 1:P
        tensor[D+p,i,j] = μ[p]
      end
    end
  end
  return tensor
end