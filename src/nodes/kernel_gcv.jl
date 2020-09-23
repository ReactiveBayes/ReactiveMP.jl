export make_node, rule, KernelGCV, KernelGCVMetadata

import LinearAlgebra: logdet, tr

struct KernelGCVMetadata
    kernelFn      :: Function
    approximation :: AbstractApproximationMethod
end

get_kernelfn(meta::KernelGCVMetadata)      = meta.kernelFn
get_approximation(meta::KernelGCVMetadata) = meta.approximation

struct KernelGCV
    meta :: KernelGCVMetadata
end

function KernelGCVNode(metadata::KernelGCVMetadata)
    return FactorNode(KernelGCV, Stochastic, ( :y, :x, :z ), ( ( 1, 2 ), ( 3, ) ), metadata)
end

function make_node(::Type{ KernelGCV }, metadata::KernelGCVMetadata, y, x, z)
    node = KernelGCVNode(metadata)
    connect!(node, :y, y)
    connect!(node, :x, x)
    connect!(node, :z, z)
    return node
end

## rules

struct FnWithApproximation
    fn            :: Function
    approximation :: AbstractApproximationMethod
end

@symmetrical function multiply_messages(m1::Message{ <: MvNormalMeanCovariance }, m2::Message{ <: FnWithApproximation })
    m2data = getdata(m2)
    
    m, V = approximate_meancov(m2data.approximation, (s) -> exp(m2data.fn(s)), getdata(m1))
    return Message(MvNormalMeanCovariance(m, PDMat(Matrix(Hermitian(V)))))
end