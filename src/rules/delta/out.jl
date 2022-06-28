using Random

# We define a rule for `DeltaFn{f}` where `f` is a callable reference to our function and can be called as `f(1, 2, 3)` blabla
# `m_ins` is a tuple of input messages
# `meta` handles reference to our meta object
# `N` can be used for dispatch and can handle special cases, e.g `m_ins::NTuple{1, NormalMeanPrecision}` means that `DeltaFn` has only 1 input

@rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} =
    begin
        message_samples = rand(meta.rng, m_in, meta.nsamples)
        return Sample
        List(map(x -> f(x...), message_samples))
    end

@rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{1, Any}, meta::LinearApproximation) where {f} = begin
    mean, var = mean(m_ins[1]), var(m_ins[1])
    (a, b) = localLinearization(g, mean)
    m = a * mean + b
    V = A * var * A'
    return Normal(m, V)
end

# @rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::LinearApproximationKnownInverse) where {f, N} =
#     begin
#         return NormalMeanPrecision(f(mean.(m_ins)...), 1.0)
#     end

# @rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::LinearApproximationUnknownInverse) where {f, N} =
#     begin
#         return NormalMeanPrecision(f(mean.(m_ins)...), 1.0)
#     end
