function collect_data(solver,op,r,args...)
  @abstractmethod
end

function collect_data(solver::NeuralSolver,op::ParamOperator,r::Realisation,args...)
  trial = get_trial(op)
  values = solve(solver,op,r,args...)
  coords = get_free_dof_coordinates(trial)
  build_dataset(solver,values,coords,r)
end

function build_dataset(solver::NeuralSolver,values,coords,r)
  build_dataset(get_sampler(solver),values,coords,r)
end