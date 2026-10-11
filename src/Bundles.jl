struct FeaturesBundle{A<:Tuple}
  bundle::A
end

FeaturesBundle(args::Function...) = FeaturesBundle(args)

function get_features(f::FeaturesBundle,V::FESpace,r::Realisation)
  Vr = V(r)
  features = zeros(num_free_dofs(Vr),length(f.bundle),num_params(r))
  @inbounds @views for (j,f) in enumerate(f.bundle)
    fh = interpolate(f,Vr)
    fv = get_free_dof_values(fh)
    for k in param_eachindex(fv)
      features[:,j,k] = param_getindex(fv,k)
    end
  end
  return features
end