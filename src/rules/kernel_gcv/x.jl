export rule

@rule KernelGCV(:x, Marginalisation) (
    m_y::MvNormalMeanCovariance,
    q_z::MvNormalMeanCovariance,
    meta::KernelGCVMetadata
) = begin
    kernelfunction = get_kernelfn(meta)
    Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> cholinv(kernelfunction(s)), q_z)
    return MvNormalMeanCovariance(mean(m_y), cov(m_y) + cholinv(Λ_out))
end

@rule KernelGCV(:x, Marginalisation) (m_y::MvNormalMeanPrecision, q_z::MvNormalMeanPrecision, meta::KernelGCVMetadata) =
    begin
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> cholinv(kernelfunction(s)), q_z)
        return MvNormalMeanPrecision(mean(m_y), cholinv(cov(m_y) + cholinv(Λ_out)))
    end
