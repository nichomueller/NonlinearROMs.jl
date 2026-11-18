using Gridap 
using GridapROMs
using GridapROMs.RBSteady
using NonlinearROMs

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = parameterise(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterise(u0,μ)

stiffness(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf

ptspace = TransientParamSpace(pdomain,tdomain)

fesolver = ThetaMethod(LUSolver(),dt,θ)
feop = TransientLinearParamFEOperator(rhs,(stiffness,mass),ptspace,trial,test)

μ = realisation(pspace;nparams=50)
μt = TransientRealisation(μ,collect(tdomain))
uhμ = solve(fesolver,feop,μt,uh0μ)

using Gridap.Arrays
using GridapROMs.ParamDataStructures

A = RolloutData(uhμ)
x = get_coords(test)

sol = A.data
np = num_params(sol)
N = length(x)
r1 = NonlinearROMs.stage_ranges(A;stage=:first)
r2 = NonlinearROMs.stage_ranges(A;stage=:second)
Δ = length(first(r1))

o = NonlinearROMs._offsets(sol,testitem(x))
D = o[end]-1

stage1 = [zeros(Float32,D*Δ,N,np) for _ in eachindex(r1)]
stage2 = [zeros(Float32,D*Δ,N,np) for _ in eachindex(r2)]

Tmax = last(A.steps)-1

np = num_params(A.data)
values,offsets = NonlinearROMs._init_data(A,x)

T = typeof(values)
s1 = Vector{T}(undef,length(r1))
s2 = Vector{T}(undef,length(r2))
@inbounds for k in eachindex(r1)
  s1[k] = similar(values)
  s2[k] = similar(values)
end

ah,state = iterate(A.data)
ax = lazy_map(ah,x)
cache = array_cache(ax)

k = 1
i = 1
axi = getindex!(cache,ax,i)
_fill_stage!(stage,axi,o,k,i,np)
# for k in 1:last(a.steps)-1
#   for i in eachindex(ax)
#     axi = getindex!(cache,ax,i)
#     _fill_stage!(stage,axi,o,k,i,np)
#   end
#   (r,ah),state = iterate(a.data,state)
#   ax = lazy_map(ah,x)
# end

stage1,stage2 = NonlinearROMs._split_stages(A,r1,r2,x)
ostage1,ostage2 = NonlinearROMs._old_split_stages(A,r1,r2,x)