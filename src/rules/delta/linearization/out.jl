
@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Linearization} = begin
    # Collect statistics with individual means and covariances
    return approximate(getmethod(meta), f, m_ins)
end
