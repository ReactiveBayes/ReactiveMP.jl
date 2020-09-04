export make_node, rule, KernelGCV, KernelGCVMetadata

import LinearAlgebra: logdet, tr

struct KernelGCVMetadata{F, A}
    kernelFn      :: F
    approximation :: A
end

get_kernelfn(meta::KernelGCVMetadata)      = meta.kernelFn
get_approximation(meta::KernelGCVMetadata) = meta.approximation

struct KernelGCV{F, A}
    meta :: KernelGCVMetadata{F, A}
end

function KernelGCVNode(metadata::KernelGCVMetadata{F, A}) where { F, A }
    return FactorNode(KernelGCV{F, A}, Stochastic, ( :x, :y, :z ), ( ( 1, 2 ), ( 3, ) ), metadata)
end

function make_node(::Type{ <: KernelGCV }, metadata::KernelGCVMetadata, x, z, y)
    node = KernelGCVNode(metadata)
    connect!(node, :x, x)
    connect!(node, :z, z)
    connect!(node, :y, y)
    return node
end

## rules

struct FnWithApproximation{F, A}
    fn            :: F
    approximation :: A
end

function rule(
    ::Type{ <: KernelGCV }, 
    ::Type{ Val{:z} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal{ <: MvNormalMeanCovariance}}, 
    meta::KernelGCVMetadata)
    ##
    q_xy = marginals[1]

    dims = Int64(ndims(q_xy) / 2)

    m_xy   = mean(q_xy)
    cov_xy = cov(q_xy)

    cov11 = @view cov_xy[1:dims,1:dims]
    cov12 = @view cov_xy[1:dims,dims+1:end]
    cov21 = @view cov_xy[dims+1:end,1:dims]
    cov22 = @view cov_xy[dims+1:end,dims+1:end]

    m1 = @view m_xy[1:dims]
    m2 = @view m_xy[dims+1:end]

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
    ::Type{ <: KernelGCV }, 
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
    ::Type{ Val{:x_y} }, 
    messages::Tuple{Message{ <: MvNormalMeanCovariance{T}},Message{<:MvNormalMeanCovariance{T}}}, 
    marginals::Tuple{Marginal{ <: MvNormalMeanCovariance{T} }}, 
    meta::KernelGCVMetadata) where { T <: Real }
    ##

    kernelfunction = get_kernelfn(meta)
    Λ = approximate_kernel_expectation(get_approximation(meta), (z) -> inv(kernelfunction(z)), marginals[1])

    Λx = inv(cov(messages[1]))
    Λy = inv(cov(messages[2]))

    wx = Λx * mean(messages[1])
    wy = Λy * mean(messages[2])

    C = inv(PDMat([ Λ + Λx -Λ; -Λ Λ + Λy ]))
    m = C * [ wx ; wy ]

    return MvNormalMeanCovariance(m, C)
end