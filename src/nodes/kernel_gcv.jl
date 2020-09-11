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

@rule(
    form        => Type{ KernelGCV },
    on          => :z,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_y_x::MvNormalMeanCovariance, ),
    meta        => KernelGCVMetadata,
    begin 
        dims = Int64(ndims(q_y_x) / 2)

        m_yx   = mean(q_y_x)
        cov_yx = cov(q_y_x)
    
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
)

# symmetric for y and x todo
@rule(
    form        => Type{ KernelGCV },
    on          => :y,
    vconstraint => Marginalisation,
    messages    => (m_x::MvNormalMeanCovariance{T}, ) where { T <: Real },
    marginals   => (q_z::MvNormalMeanCovariance{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> inv(kernelfunction(s)), q_z)

        return MvNormalMeanCovariance(mean(m_x), cov(m_x) + inv(PDMat(Λ_out)))
    end
)

@rule(
    form        => Type{ KernelGCV },
    on          => :x,
    vconstraint => Marginalisation,
    messages    => (m_y::MvNormalMeanCovariance{T}, ) where { T <: Real },
    marginals   => (q_z::MvNormalMeanCovariance{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> inv(kernelfunction(s)), q_z)

        return MvNormalMeanCovariance(mean(m_y), cov(m_y) + inv(PDMat(Λ_out)))
    end
)

@symmetrical function multiply_messages(m1::Message{ <: MvNormalMeanCovariance }, m2::Message{ <: FnWithApproximation })
    m2data = getdata(m2)
    m, V = approximate_meancov(m2data.approximation, (s) -> exp(m2data.fn(s)), getdata(m1))
    return Message(MvNormalMeanCovariance(m, PDMat(V)))
end

## marginal rules

@marginalrule(
    form      => Type{ <: KernelGCV },
    on        => :y_x,
    messages  => (m_y::MvNormalMeanCovariance{T}, m_x::MvNormalMeanCovariance{T}) where { T <: Real },
    marginals => (q_z::MvNormalMeanCovariance{T}, ),
    meta      => KernelGCVMetadata,
    begin
        kernelfunction = get_kernelfn(meta)
        Λ = approximate_kernel_expectation(get_approximation(meta), (z) -> inv(kernelfunction(z)), q_z)

        Λy = inv(cov(m_y))
        Λx = inv(cov(m_x))

        wy = Λy * mean(m_y)
        wx = Λx * mean(m_x)

        C = inv(PDMat([ Λ + Λy -Λ; -Λ Λ + Λx ]))
        m = C * [ wy ; wx ]

        return MvNormalMeanCovariance(m, C)
    end
)