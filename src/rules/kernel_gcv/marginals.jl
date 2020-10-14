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

        C = inv(PDMat(Matrix(Hermitian([ Λ + Λy -Λ; -Λ Λ + Λx ]))))
        m = C * [ wy ; wx ]

        return MvNormalMeanCovariance(m, C)
    end
)

@marginalrule(
    form      => Type{ <: KernelGCV },
    on        => :y_x,
    messages  => (m_y::MvNormalMeanPrecision{T}, m_x::MvNormalMeanPrecision{T}) where { T <: Real },
    marginals => (q_z::MvNormalMeanPrecision{T}, ),
    meta      => KernelGCVMetadata,
    begin
        kernelfunction = get_kernelfn(meta)
        C = approximate_kernel_expectation(get_approximation(meta), (z) -> inv(kernelfunction(z)), q_z)

        Cy = precision(m_y)
        Cx = precision(m_x)

        wy = Cy * mean(m_y)
        wx = Cx * mean(m_x)

        Λ = PDMat(Matrix(Hermitian([ C + Cy -C; -C C + Cx ])))
        μ = inv(Λ) * [ wy ; wx ]

        return MvNormalMeanPrecision(μ, Λ)
    end
)