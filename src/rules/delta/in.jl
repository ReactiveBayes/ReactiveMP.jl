using Random

@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::SamplingApproximation) where {f} = begin
    return rand(meta.rng, NormalMeanPrecision(0.0, 1.0), meta.n)
end
