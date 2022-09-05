# known inverse, single input
@rule DeltaFn{f}((:in, _), Marginalisation) (
    m_out::Any,
    m_ins::Nothing,
    meta::DeltaExtended{T}
) where {f, T <: Function} =
    begin
        @show "in"
        μ_out, Σ_out = mean_cov(m_out)
        (A, b) = localLinearizationSingleIn(meta.inverse, μ_out)
        m = A * μ_out + b
        V = A * Σ_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# I expect this rule to be called when inverse is given
@rule DeltaFn{f}((:in, k), Marginalisation) (m_out::Any, m_ins::Any, meta::DeltaExtended{T}) where {f, T} =
    begin
        @show "multiple input backward"
        (ms, Vs) = collectStatistics(m_in, q_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearization(meta.inverse, ms)
        (mc, Vc) = concatenateGaussianMV(ms, Vs)
        m = A * m_out + b
        V = A * V_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

# why this method is called?
@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T} =
    begin
        @show meta.inverse[k](mean(q_ins))
        (A, b) = localLinearizationMultiIn(meta.inverse[k], mean(q_ins))
        (mc, Vc) = concatenateGaussianMV(ms, Vs)
        m = A * m_out + b
        V = A * V_out * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate

        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where {f, T <: Nothing} =
    begin
        return MvNormalMeanPrecision(zeros(2), diageye(2))
    end
