export marginalrule

@marginalrule NonLinear(:x) (m_out::NormalDistributionsFamily, m_x::NormalDistributionsFamily, meta::NonLinearMetadata) = begin
    return NormalMeanPrecision(mean(m_x), precision(m_x))
end