@marginalrule GaussianProcess(:out_params) (m_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::Any) = begin
    return (out = m_out.finitemarginal, params = m_params)
end