export make_node, rule, KernelGCV, KernelGCVMetadata

import LinearAlgebra: logdet, tr

struct KernelGCVMetadata{F, A}
    kernelFn      :: F
    approximation :: A
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

struct FnWithApproximation{F, A}
    fn            :: F
    approximation :: A
end

function prod(::ProdPreserveParametrisation, left::MvNormalMeanCovariance, right::FnWithApproximation)
    μ, Σ = approximate_meancov(m2data.approximation, (s) -> exp(right.fn(s)), left)
    return MvNormalMeanCovariance(μ, Σ)
end

function prod(::ProdPreserveParametrisation, left::FnWithApproximation, right::MvNormalMeanCovariance)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::MvNormalMeanPrecision, right::FnWithApproximation)
    μ, Σ = approximate_meancov(m2data.approximation, (s) -> exp(right.fn(s)), left)
    return MvNormalMeanPrecision(μ, cholinv(Σ))
end

function prod(::ProdPreserveParametrisation, left::FnWithApproximation, right::MvNormalMeanPrecision)
    return prod(ProdPreserveParametrisation(), right, left)
end