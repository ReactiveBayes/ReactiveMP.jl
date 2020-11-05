@rule(
    formtype    => KernelGCV,
    on          => :x,
    vconstraint => Marginalisation,
    messages    => (m_y::MvNormalMeanCovariance{T}, ) where { T },
    marginals   => (q_z::MvNormalMeanCovariance{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> cholinv(kernelfunction(s)), q_z)

        return MvNormalMeanCovariance(mean(m_y), cov(m_y) + cholinv(Λ_out))
    end
)

@rule(
    formtype    => KernelGCV,
    on          => :x,
    vconstraint => Marginalisation,
    messages    => (m_y::MvNormalMeanPrecision{T}, ) where { T },
    marginals   => (q_z::MvNormalMeanPrecision{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> cholinv(kernelfunction(s)), q_z)

        return MvNormalMeanPrecision(mean(m_y), cholinv(cov(m_y) + cholinv(Λ_out)))
    end
)