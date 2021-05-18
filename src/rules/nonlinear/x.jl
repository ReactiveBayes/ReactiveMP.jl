export rule

@rule NonLinear(:x, Marginalisation) (m_out::NormalDistributionsFamily, meta::NonLinearMetadata) = begin
    
    mm_out = meta.fi(mean(m_out))
    mw_out = precision(m_out) / meta.dfi(mean(m_out))^2

    return NormalMeanPrecision(mm_out, mw_out)

end