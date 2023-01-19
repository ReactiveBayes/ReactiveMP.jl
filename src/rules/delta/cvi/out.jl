
@rule DeltaFn(:out, Marginalisation) (q_ins::FactorizedJoint{P}, meta::DeltaMeta{M}) where {P <: NTuple{1}, M <: CVI} = begin
    method            = getmethod(meta)
    q_sample_friendly = ReactiveMP.logpdf_sample_friendly(q_ins[1])[2]
    rng               = something(method.rng, Random.GLOBAL_RNG)
    samples           = map(x -> rand(rng, q_sample_friendly), 1:(method.n_samples))
    q_out             = map(getnodefn(Val(:out)), samples)
    return ProdFinal(SampleList(q_out))
end

@rule DeltaFn(:out, Marginalisation) (q_ins::FactorizedJoint, meta::DeltaMeta{M}) where {M <: CVI} = begin
    method = getmethod(meta)
    q_ins_sample_friendly = map(marginal -> ReactiveMP.logpdf_sample_friendly(marginal)[2], getmultipliers(q_ins))
    rng = something(method.rng, Random.GLOBAL_RNG)
    q_ins_samples = map(marginal -> rand(rng, marginal, method.n_samples), q_ins_sample_friendly)
    samples_linear = map(cvilinearize, q_ins_samples)
    g = getnodefn(Val(:out))
    samples = map(x -> g(x...), zip(samples_linear...))
    return ProdFinal(SampleList(samples))
end


### gp test 
@rule DeltaFn(:out, Marginalisation) (q_ins::FactorizedJoint, meta::Tuple{<:ProcessMeta, <:DeltaMeta{M}}) where {M <: CVI} = begin 
    method            = getmethod(meta[2])
    q_sample_friendly = ReactiveMP.logpdf_sample_friendly(q_ins[1])[2]
    rng               = something(method.rng, Random.GLOBAL_RNG)
    samples           = map(x -> rand(rng, q_sample_friendly), 1:(method.n_samples))
    q_out             = map(getnodefn(Val(:out)), samples)
    return ProdFinal(SampleList(q_out))
end