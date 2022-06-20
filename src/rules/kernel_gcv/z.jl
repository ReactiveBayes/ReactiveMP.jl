export rule

@rule KernelGCV(:z, Marginalisation) (q_y_x::MvNormalMeanCovariance, meta::KernelGCVMetadata) = begin
    dims = Int64(ndims(q_y_x) / 2)

    m_yx   = mean(q_y_x)
    cov_yx = cov(q_y_x)

    cov11 = @view cov_yx[1:dims, 1:dims]
    cov12 = @view cov_yx[1:dims, dims+1:end]
    cov21 = @view cov_yx[dims+1:end, 1:dims]
    cov22 = @view cov_yx[dims+1:end, dims+1:end]

    m1 = @view m_yx[1:dims]
    m2 = @view m_yx[dims+1:end]

    psi = cov11 + cov22 - cov12 - cov21 + (m1 - m2) * (m1 - m2)'

    kernelfunction = get_kernelfn(meta)

    logpdf = (z) -> begin
        gz = kernelfunction(z)
        return -0.5 * (logdet(gz) + tr(cholinv(gz) * psi))
    end

    return FnWithApproximation(logpdf, get_approximation(meta))
end

@rule KernelGCV(:z, Marginalisation) (q_y_x::MvNormalMeanPrecision, meta::KernelGCVMetadata) = begin
    dims = Int64(ndims(q_y_x) / 2)

    m_yx   = mean(q_y_x)
    cov_yx = cov(q_y_x)

    cov11 = @view cov_yx[1:dims, 1:dims]
    cov12 = @view cov_yx[1:dims, dims+1:end]
    cov21 = @view cov_yx[dims+1:end, 1:dims]
    cov22 = @view cov_yx[dims+1:end, dims+1:end]

    m1 = @view m_yx[1:dims]
    m2 = @view m_yx[dims+1:end]

    psi = cov11 + cov22 - cov12 - cov21 + (m1 - m2) * (m1 - m2)'

    kernelfunction = get_kernelfn(meta)

    logpdf = (z) -> begin
        gz = kernelfunction(z)
        return -0.5 * (logdet(gz) + tr(cholinv(gz) * psi))
    end

    return FnWithApproximation(logpdf, get_approximation(meta))
end
