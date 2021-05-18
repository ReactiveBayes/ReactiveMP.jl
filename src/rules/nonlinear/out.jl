export rule

@rule NonLinear(:out, Marginalisation) (m_x::NormalDistributionsFamily, meta::NonLinearMetadata) = begin
    
    m_out = meta.f(mean(m_x))
    w_out = precision(m_x) / meta.df(mean(m_x))^2

    return NormalMeanPrecision(m_out, w_out)

end
