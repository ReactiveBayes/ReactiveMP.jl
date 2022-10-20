
@rule DeltaFn{f}(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {f, N, M <: Unscented} = begin
    return approximate(getmethod(meta), f, m_ins)
end
