
@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::JointNormal, m_in::NormalDistributionsFamily, meta::DeltaMeta{M, Nothing}) where {f, M <: Linearization} = begin
    # we need dimension of the interface variables
    # Marginalize joint belief on in's
    inx = k
    q_ins, ds = q_ins.dist, q_ins.ds
    μ_in, Σ_in = mean_cov(q_ins)

    (μ_inx, Σ_inx) = marginalizeGaussianMV(μ_in, Σ_in, ds, inx)
    Λ_inx = cholinv(Σ_inx) # Convert to canonical statistics
    ξ_inx = Λ_inx * μ_inx

    F = variate_form(m_in)

    ξ_inx, Λ_inx = F == Univariate ? (first(ξ_inx), first(Λ_inx)) : (ξ_inx, Λ_inx)

    # Divide marginal on inx by forward message
    (ξ_fw_inx, Λ_fw_inx) = weightedmean_precision(m_in)

    ξ_bw_inx = ξ_inx - ξ_fw_inx
    Λ_bw_inx = Λ_inx - Λ_fw_inx # Note: subtraction might lead to posdef violations

    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ_bw_inx, Λ_bw_inx)
end

@rule DeltaFn{f}((:in, _), Marginalisation) (m_out::NormalDistributionsFamily, m_ins::Nothing, meta::DeltaMeta{M, I}) where {f, M <: Linearization, I <: Function} = begin
    μ_out, Σ_out = mean_cov(m_out)
    (A, b) = localLinearizationSingleIn(getinverse(meta), μ_out)
    m = A * μ_out + b
    V = A * Σ_out * A'
    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end

@rule DeltaFn{f}((:in, k), Marginalisation) (m_out::NormalDistributionsFamily, m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M, I}) where {f, N, M <: Linearization, L, I <: NTuple{L, Function}} = begin
    (μs_in, Σs_in) = collectStatistics(m_out, m_ins...)
    (A, b) = localLinearizationMultiIn(getinverse(meta, k), μs_in)
    (μ_in, Σ_in, _) = concatenateGaussianMV(μs_in, Σs_in)
    m = A * μ_in + b
    V = A * Σ_in * A'
    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end


