export KernelGCV, KernelGCVMetadata

import LinearAlgebra: logdet, tr

struct KernelGCVMetadata{F, A}
    kernelFn      :: F
    approximation :: A
end

get_kernelfn(meta::KernelGCVMetadata)      = meta.kernelFn
get_approximation(meta::KernelGCVMetadata) = meta.approximation

struct KernelGCV end

@node KernelGCV Stochastic [y, x, z]

# TODO: Remove in favor of Generic Functional Message
struct FnWithApproximation{F, A}
    fn            :: F
    approximation :: A
end

prod_analytical_rule(::Type{<:MultivariateNormalDistributionsFamily}, ::Type{<:FnWithApproximation}) =
    ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::MultivariateNormalDistributionsFamily, right::FnWithApproximation)
    μ, Σ = approximate_meancov(right.approximation, (s) -> exp(right.fn(s)), left)
    return MvNormalMeanCovariance(μ, Σ)
end

prod_analytical_rule(::Type{<:FnWithApproximation}, ::Type{<:MultivariateNormalDistributionsFamily}) =
    ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::FnWithApproximation, right::MultivariateNormalDistributionsFamily)
    return prod(ProdAnalytical(), right, left)
end
