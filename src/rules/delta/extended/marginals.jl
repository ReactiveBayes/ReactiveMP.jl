export marginalrule

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaExtended) where { f, N } = begin 
    return MvNormalMeanPrecision(zeros(N), diageye(N))
end

@marginalrule DeltaFn{f}(:ins) (m_out::Any, m_ins::NTuple{1, Any}, meta::DeltaExtended) where { f } = begin 
    return NormalMeanPrecision(0.0, 1.0)
end