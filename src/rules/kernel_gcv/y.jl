@rule(
    form        => Type{ KernelGCV },
    on          => :y,
    vconstraint => Marginalisation,
    messages    => (m_x::MvNormalMeanCovariance{T}, ) where { T <: Real },
    marginals   => (q_z::MvNormalMeanCovariance{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> cholinv(kernelfunction(s)), q_z)

        return MvNormalMeanCovariance(mean(m_x), cov(m_x) + cholinv(Λ_out))
    end
)

@rule(
    form        => Type{ KernelGCV },
    on          => :y,
    vconstraint => Marginalisation,
    messages    => (m_x::MvNormalMeanPrecision{T}, ) where { T <: Real },
    marginals   => (q_z::MvNormalMeanPrecision{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> inv(kernelfunction(s)), q_z)

        return MvNormalMeanPrecision(mean(m_x), cholinv(cov(m_x) + cholinv(Λ_out)))
    end
)