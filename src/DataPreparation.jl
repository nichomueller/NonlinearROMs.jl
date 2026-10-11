# steady eqs 

# Generic: (x,{ah^{k,p}}_{k=1}^{K,P},μh^p})_{p=1}^P

# GNOs & HAMLET: (V,E) + scatter_on_graph((x,{ah^{k,p}}_{k=1}^{K,P},μh^p})_{p=1}^P)

# DeepONet: 
# - branch: ({ah_{sensor}^{k,p}}_{k=1}^{K},μh^p})_{p=1}^P (i.e. interpolation on subgrids)
# - trunk: (x,μh^p})_{p=1}^P

# NOMADS
# - encoder {z^p}_{p=1}^P = E(({ah^{k,p}}_{k=1}^{K})_{p=1}^P)
# - decoder (x,z^p,μh^p)_{p=1}^P

# transient eqs

# Generic: (x,{ah^{k,n,p}}_{k=1}^{K,N,P},{μh^p}})_{n,p=1}^P

# GNOs & HAMLET: (V,E) + scatter_on_graph((x,{ah^{k,n,p}}_{k=1}^{K,P},{μh^p}})_{p=1}^P)

# DeepONet: 
# - branch: ({ah_{sensor}^{k,n,p}}_{k=1}^{K},{μh^p}})_{p=1}^P (i.e. interpolation on subgrids)
# - trunk: (x,μh^{p})_{p=1}^P

# NOMADS
# - encoder {z^{n,p}}_{n,p=1}^{N,P} = E(({ah^{k,n,p}}_{k,n=1}^{K,N})_{p=1}^P)
# - decoder (x,{z^p},μh^{p}})_{p=1}^P

export NeuralModel
export FiniteDimensionalModel
export CoordinateNeuralModel
export IntegralKernel
export DeepONet
export NOMAD
export MultiLayerPerceptron
export AutoEncoder
export VariationalAutoEncoder
export AutoDecoder
export build_model
export KernelNeuralModel

function ()
  
end