@marginalrule DeltaFn{f}(:ins) (q_out::Any, m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} = begin
    return MvNormalMeanPrecision(zeros(N), diageye(N))
end
