using Random
using ReactiveMP

# We define a rule for `DeltaFn{f}` where `f` is a callable reference to our function and can be called as `f(1, 2, 3)` blabla
# `m_ins` is a tuple of input messages
# `meta` handles reference to our meta object
# `N` can be used for dispatch and can handle special cases, e.g `m_ins::NTuple{1, NormalMeanPrecision}` means that `DeltaFn` has only 1 input

@rule DeltaFn{f}(:out, Marginalisation) (q_ins::FactorizedJoint{P}, meta::DeltaMeta{M}) where {f, P <: NTuple{1}, M <: CVIApproximation} = begin
    method = getmethod(meta)
    q_sample_friendly = ReactiveMP.logpdf_sample_friendly(q_ins[1])[2]
    rng               = something(method.rng, Random.GLOBAL_RNG)
    samples           = map(x -> rand(rng, q_sample_friendly), 1:(method.n_samples))
    q_out             = map(f, samples)
    return ProdFinal(SampleList(q_out))
end

@rule DeltaFn{f}(:out, Marginalisation) (q_ins::FactorizedJoint, meta::DeltaMeta{M}) where {f, M <: CVIApproximation} = begin
    method = getmethod(meta)
    q_ins_sample_friendly = map(marginal -> ReactiveMP.logpdf_sample_friendly(marginal)[2], getmultipliers(q_ins))
    rng = something(method.rng, Random.GLOBAL_RNG)
    q_ins_samples = map(marginal -> rand(rng, marginal, method.n_samples), q_ins_sample_friendly)
    samples_linear = map(cvilinearize, q_ins_samples)
    samples = map(x -> f(x...), zip(samples_linear...))
    return ProdFinal(SampleList(samples))
end
