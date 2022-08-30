@rule DeltaFn{f}((:in, k), Marginalisation) (q_ins::Any, m_in::Any, meta::DeltaExtended{T}) where { f, T<:Function } = begin 
    return NormalMeanPrecision(0.0, 1.0)
end