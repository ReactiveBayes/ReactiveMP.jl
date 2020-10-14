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

        return MvNormalMeanCovariance(mean(m_y), cov(m_y) + inv(PDMat(Matrix(Hermitian(Λ_out)))))
    end
)

@rule(
    form        => Type{ KernelGCV },
    on          => :x,
    vconstraint => Marginalisation,
    messages    => (m_y::MvNormalMeanPrecision{T}, ) where { T <: Real },
    marginals   => (q_z::MvNormalMeanPrecision{T}, ),
    meta        => KernelGCVMetadata,
    begin 
        kernelfunction = get_kernelfn(meta)
        Λ_out = approximate_kernel_expectation(get_approximation(meta), (s) -> inv(kernelfunction(s)), q_z)

        return MvNormalMeanPrecision(mean(m_y), inv(cov(m_y) + inv(PDMat(Matrix(Hermitian(Λ_out))))))
    end
)