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

function rule(
    ::Type{ KernelGCV }, 
    ::Type{ Val{:z} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal{ <: MvNormalMeanCovariance }}, 
    meta::KernelGCVMetadata)
    ##
    q_yx = marginals[1]

    dims = Int64(ndims(q_yx) / 2)

    m_yx   = mean(q_yx)
    cov_yx = cov(q_yx)

    cov11 = @view cov_yx[1:dims,1:dims]
    cov12 = @view cov_yx[1:dims,dims+1:end]
    cov21 = @view cov_yx[dims+1:end,1:dims]
    cov22 = @view cov_yx[dims+1:end,dims+1:end]

    m1 = @view m_yx[1:dims]
    m2 = @view m_yx[dims+1:end]

    psi = cov11 + cov22 - cov12 - cov21 + (m1 - m2)*(m1 - m2)'

    kernelfunction = get_kernelfn(meta)

    logpdf = (z) -> begin
        gz = kernelfunction(z)
        return -0.5*(logdet(gz) + tr(inv(gz)*psi))
    end

    return FnWithApproximation(logpdf, get_approximation(meta))
end

# symmetric for y and x todo
function rule(
    ::Type{ KernelGCV }, 
    ::Union{ Type{Val{:y}}, Type{Val{:x}} }, 
    ::Marginalisation, 
    messages::Tuple{Message{ <: MvNormalMeanCovariance{T}}}, 
    marginals::Tuple{Marginal{ <: MvNormalMeanCovariance{T}}},
    meta::KernelGCVMetadata) where { T <: Real }
    ##
    mean_m, cov_m = mean(messages[1]), cov(messages[1])

    kernelfunction = get_kernelfn(meta)
    Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> inv(kernelfunction(s)), marginals[1])

    return MvNormalMeanCovariance(mean_m, cov_m + inv(PDMat(Λ_out)))
end

@symmetrical function multiply_messages(m1::Message{ <: MvNormalMeanCovariance }, m2::Message{ <: FnWithApproximation })
    m2data = getdata(m2)
    m, V = approximate_meancov(m2data.approximation, (s) -> exp(m2data.fn(s)), getdata(m1))
    return Message(MvNormalMeanCovariance(m, PDMat(V)))
end

function marginalrule(
    ::Type{ <: KernelGCV }, 
    ::Type{ Val{:y_x} }, 
    messages::Tuple{Message{ <: MvNormalMeanCovariance{T}},Message{<:MvNormalMeanCovariance{T}}}, 
    marginals::Tuple{Marginal{ <: MvNormalMeanCovariance{T} }}, 
    meta::KernelGCVMetadata) where { T <: Real }
    ##

    kernelfunction = get_kernelfn(meta)
    Λ = approximate_kernel_expectation(get_approximation(meta), (z) -> inv(kernelfunction(z)), marginals[1])

    Λy = inv(cov(messages[1]))
    Λx = inv(cov(messages[2]))

    wy = Λy * mean(messages[1])
    wx = Λx * mean(messages[2])

    C = inv(PDMat([ Λ + Λy -Λ; -Λ Λ + Λx ]))
    m = C * [ wy ; wx ]

    return MvNormalMeanCovariance(m, C)
end