function Base.collect(sol::TransientParamFESolution)
  r = sol.odesol.r
  vals = collect(sol.odesol)
  trial = sol.trial(r)
  FEFunction(trial,vals)
end

struct RolloutData{A,B<:AbstractVector,C<:Union{Tuple,AbstractVector}}
  data::A
  times::B
  steps::C
end

function RolloutData(data,times::AbstractVector,step::Int=2)
  @check step % 2 == 0 "Step must be even"
  _steps = 1:step:length(times)
  steps = last(_steps) == length(times) ? _steps : vcat(_steps,length(times))
  RolloutData(data,times,steps)
end

function Base.collect(a::RolloutData)
  data = collect(a.data)
  T = typeof(data)
  rollout = Vector{T}(undef,length(a.steps)-1)
  for k in eachindex(rollout)
    rollout[k] = data[a.steps[k]:a.steps[k+1]]
  end
  return rollout
end

function Base.collect(a::RolloutData{<:Function})
  map(1:length(a.steps)-1) do k
    [a.data(t) for t in a.times[a.steps[k]]:a.times[a.steps[k+1]]]
  end
end

function InputSpace(x::AbstractVector{<:Point},a::RolloutData)
  data = collect(a)
  item = first(data)
  coords = _get_coords(x,num_params(item))
  features = _get_features(data,x)
  InputSpace(coords,features)
end

function InputSpace(x::AbstractVector{<:Point},a::RolloutData...)
  spaces = map(ai -> InputSpace(x,ai),a)
  coords = cat(map(get_coords,spaces)...;dims=3)
  features = cat(map(get_features,spaces)...;dims=3)
  InputSpace(coords,features)
end

function OutputSpace()
  
end

# function rollout_data(
#   fspace,
#   features_fn::Function,
#   times::AbstractVector,
#   states::AbstractVector;
#   noise_std::Real=TData(0.01),
#   rng::AbstractRNG=Random.default_rng()
#   )

#   Nt = length(times)
#   @check Nt == length(states) "Need exactly one state per time"
#   @check Nt > 1 "Need at least 2 time steps to form a consecutive pair"

#   x = get_coords(fspace)
#   rid1 = _get_values(states[1],x)
#   ranges = rid1.ranges
#   point_ranges = rid1.point_ranges

#   ins = Vector{AbstractInputData}(undef,Nt-1)
#   outs = Vector{AbstractInputData}(undef,Nt-1)
#   for i in 1:Nt-1
#     feat_i = _get_features(features_fn(times[i]),x)
#     u_i = _get_values(states[i],x).data
#     u_ip1 = _get_values(states[i+1],x).data
#     if !iszero(noise_std)
#       u_i = u_i .+ TData(noise_std).*maximum(abs,u_i).*TData.(randn(rng,size(u_i)))
#     end
#     ins[i] = vcat(feat_i,u_i)
#     outs[i] = u_ip1 .- u_i
#   end

#   input = cat(ins...;dims=3)
#   output = cat(outs...;dims=3)
#   coords = _get_coords(x,size(input,3))

#   input_space = InputSpace(coords,input)
#   output_space = OutputSpace(get_fe_space(states[1]),InputData(output,ranges,point_ranges))
#   return input_space,output_space
# end

# function rollout(
#   solver::NeuralSolver,
#   op::NeuralOperator,
#   fspace,
#   features_fn::Function,
#   times::AbstractVector,
#   u0::AbstractInputData
#   )

#   x = get_coords(fspace)
#   K = size(u0,3)
#   states = Vector{AbstractInputData}(undef,length(times))
#   states[1] = u0
#   u = u0
#   for i in 1:length(times)-1
#     feat_i = _get_features(features_fn(times[i]),x)
#     @check size(feat_i,3) == K "features_fn(t) must be evaluated at the same K parameters as u0"
#     input_i = InputSpace(_get_coords(x,K),vcat(feat_i,u))
#     du = solve(solver,op,input_i)
#     u = u .+ du
#     states[i+1] = u
#   end
#   return states
# end

# function rollout(solver::NeuralSolver,op::NeuralOperator,fspace,features_fn::Function,times,u0::ParamFEFunction)
#   x = get_coords(fspace)
#   u0_vals = _get_values(u0,x).data
#   rollout(solver,op,fspace,features_fn,times,u0_vals)
# end

# utils

function _get_features(a::AbstractVector{<:FeatureBundle},x::AbstractVector{<:Point})
  #TODO
end