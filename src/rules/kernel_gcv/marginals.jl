export marginalrule

@marginalrule KernelGCV(:y_x) (m_y::MvNormalMeanCovariance, m_x::MvNormalMeanCovariance, q_z::MvNormalMeanCovariance, meta::KernelGCVMetadata) = begin
    kernelfunction = get_kernelfn(meta)
    Λ = approximate_kernel_expectation(get_approximation(meta), (z) -> cholinv(kernelfunction(z)), q_z)

    Λy = invcov(m_y)
    Λx = invcov(m_x)

    wy = Λy * mean(m_y)
    wx = Λx * mean(m_x)

    C = cholinv([ Λ + Λy -Λ; -Λ Λ + Λx ])
    m = C * [ wy ; wx ]

    return MvNormalMeanCovariance(m, C)  
end

@marginalrule KernelGCV(:y_x) (m_y::MvNormalMeanPrecision, m_x::MvNormalMeanPrecision, q_z::MvNormalMeanPrecision, meta::KernelGCVMetadata) = begin
    kernelfunction = get_kernelfn(meta)
    C = approximate_kernel_expectation(get_approximation(meta), (z) -> cholinv(kernelfunction(z)), q_z)

    Cy = invcov(m_y)
    Cx = invcov(m_x)

    wy = Cy * mean(m_y)
    wx = Cx * mean(m_x)

    Λ = [ C + Cy -C; -C C + Cx ]
    μ = cholinv(Λ) * [ wy ; wx ]

    return MvNormalMeanPrecision(μ, Λ)
end