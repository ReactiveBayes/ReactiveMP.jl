using Random

# We define a rule for `DeltaFn{f}` where `f` is a callable reference to our function and can be called as `f(1, 2, 3)` blabla
# `m_ins` is a tuple of input messages
# `meta` handles reference to our meta object
# `N` can be used for dispatch and can handle special cases, e.g `m_ins::NTuple{1, NormalMeanPrecision}` means that `DeltaFn` has only 1 input

@rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} =
    begin
        message_samples = rand(meta.rng, m_in, meta.nsamples)
        return SampleList(map(x -> f(x...), message_samples))
    end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{1, Any}, meta::LinearApproximation) where {f} = begin
    m1, v1 = mean(m_ins[1]), var(m_ins[1])
    (a, b) = localLinearization(f, m1)
    m = a * m1 + b
    V = a * v1 * a'
    return NormalMeanVariance(m, V)
end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::Any, meta::LinearApproximation) where {f} = begin
    mean, var = mean(m_ins[1]), var(m_ins[1])
    (a, b) = localLinearization(g, mean)
    m = a * mean + b
    V = a * var * a'
    return NormalMeanVariance(m, V)
end

@rule DeltaFn{f}(:out, Marginalisation) (q_ins::FactorProduct{P}, meta::CVIApproximation) where {f, P <: NTuple{1}} =
    begin
        q_sample_friendly = logpdf_sample_friendly(q_ins[1])[2]
        rng               = something(meta.rng, Random.GLOBAL_RNG)
        samples           = rand(rng, q_sample_friendly, meta.n_samples)
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
