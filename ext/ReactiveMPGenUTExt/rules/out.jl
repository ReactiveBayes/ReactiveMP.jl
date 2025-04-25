@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: GenUnscented} = begin
    return approximate(AsDistribution(NormalDistributionsFamily), ReactiveMP.getmethod(meta), getnodefn(meta, Val(:out)), m_ins)
end