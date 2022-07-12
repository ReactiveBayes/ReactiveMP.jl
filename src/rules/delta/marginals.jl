@marginalrule DeltaFn{f}(:ins) (q_out::Any, m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} = begin
    return MvNormalMeanPrecision(zeros(N), diageye(N))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{1, Any}, meta::CVIApproximation) where {f} = begin
end
