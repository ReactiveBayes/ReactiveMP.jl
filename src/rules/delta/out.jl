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

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{1, Any}, meta::CVIApproximation) where {f, N} = begin
    q_marginal = meta.q_ins_marginal[1]
    q_sample_friendly = logpdf_sample_friendly(q_marginal)[2]
    samples = f.(rand(q_sample_friendly, meta.n_samples))
    return ProdFinal(SampleList(samples))
end
