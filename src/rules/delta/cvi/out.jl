
@rule DeltaFn(:out, Marginalisation) (q_ins::FactorizedJoint{P}, meta::DeltaMeta{M}) where {P <: NTuple{1}, M <: CVI} = begin
    method            = ReactiveMP.getmethod(meta)
    q_sample_friendly = sampling_optimized(q_ins[1])
    rng               = something(method.rng, Random.default_rng())
    samples           = map(x -> rand(rng, q_sample_friendly), 1:(method.n_samples))
    q_out             = map(getnodefn(meta, Val(:out)), samples)
    return TerminalProdArgument(SampleList(q_out))
end

@rule DeltaFn(:out, Marginalisation) (q_ins::FactorizedJoint, meta::DeltaMeta{M}) where {M <: CVI} = begin
    method = ReactiveMP.getmethod(meta)
    q_ins_sample_friendly = map(marginal -> sampling_optimized(marginal), components(q_ins))
    rng = something(method.rng, Random.default_rng())
    q_ins_samples = map(marginal -> rand(rng, marginal, method.n_samples), q_ins_sample_friendly)
    samples_linear = map(cvilinearize, q_ins_samples)
    g = getnodefn(meta, Val(:out))
    samples = map(x -> g(x...), zip(samples_linear...))
    return TerminalProdArgument(SampleList(samples))
end
