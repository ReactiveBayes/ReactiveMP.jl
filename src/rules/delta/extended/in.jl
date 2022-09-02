

@rule DeltaFn{f}((:in, _), Marginalisation) (m_out::Any, m_ins::Nothing, meta::DeltaExtended{T}) where {f, T <: Function} =
    begin
        # (ms, Vs) = collectStatistics(m_out) # Returns arrays with individual means and covariances
        mean_out, cov_out = mean_cov(m_out)
        (A, b) = localLinearization(meta.inverse, mean_out)
        m = A*mean_out + b
        V = A*cov_out*A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T <: Function} =
    begin
        @show "hui1"
        (ms, Vs) = collectStatistics(m_in, q_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearization(meta.inverse, ms)
        (mc, Vc) = concatenateGaussianMV(ms, Vs)
        m = A*m_out + b
        V = A*V_out*A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T <: Nothing} =
    begin
        @show "hui2"
        return MvNormalMeanPrecision(zeros(2), diageye(2))
    end

