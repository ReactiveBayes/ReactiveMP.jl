
@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: Linearization} = begin
    return approximate(getmethod(meta), getnodefn(meta, Val(:out)), m_ins)
end
