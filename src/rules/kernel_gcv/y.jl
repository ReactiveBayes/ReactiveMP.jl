export rule

@rule KernelGCV(:y, Marginalisation) (m_x::MvNormalMeanCovariance, q_z::MvNormalMeanCovariance, meta::KernelGCVMetadata) = begin
    kernelfunction = get_kernelfn(meta)
    Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> cholinv(kernelfunction(s)), q_z)
    return MvNormalMeanCovariance(mean(m_x), cov(m_x) + cholinv(Λ_out))
end

@rule KernelGCV(:y, Marginalisation) (m_x::MvNormalMeanPrecision, q_z::MvNormalMeanPrecision, meta::KernelGCVMetadata) = begin
    kernelfunction = get_kernelfn(meta)
    Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> inv(kernelfunction(s)), q_z)
    return MvNormalMeanPrecision(mean(m_x), cholinv(cov(m_x) + cholinv(Λ_out)))
end
