using Random

# We define a rule for `DeltaFn{f}` where `f` is a callable reference to our function and can be called as `f(1, 2, 3)` blabla
# `m_ins` is a tuple of input messages
# `meta` handles reference to our meta object
# `N` can be used for dispatch and can handle special cases, e.g `m_ins::NTuple{1, NormalMeanPrecision}` means that `DeltaFn` has only 1 input

@rule DeltaFn{f}(:out, Marginalisation) (q_ins::FactorProduct{P}, meta::CVIApproximation) where {f, P <: NTuple{1}} =
    begin
        q_sample_friendly = logpdf_sample_friendly(q_ins[1])[2]
        rng               = something(meta.rng, Random.GLOBAL_RNG)
        samples           = map(x -> rand(rng, q_sample_friendly), 1:meta.n_samples)
        q_out             = map(f, samples)
        return ProdFinal(SampleList(q_out))
    end

@rule DeltaFn{f}(:out, Marginalisation) (q_ins::FactorProduct{NTuple{N, Any}}, meta::CVIApproximation) where {f, N} =
    begin
        q_ins_sample_friendly = [logpdf_sample_friendly(q)[2] for q in q_ins]
        rng = something(meta.rng, Random.GLOBAL_RNG)
        q_ins_samples = map(marginal -> rand(rng, q_sample_friendly, meta.n_samples), q_ins_sample_friendly)
        samples = map(x -> f(x...), zip(q_ins_samples))
        return ProdFinal(SampleList(samples))
    end
