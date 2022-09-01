@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T <: Function} =
    begin
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
        return MvNormalMeanPrecision(zeros(2), diageye(2))
    end

