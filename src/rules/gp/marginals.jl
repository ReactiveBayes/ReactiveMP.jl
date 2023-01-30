@marginalrule GaussianProcess(:out_params) (m_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, m_params::Any) = begin
    return (out = m_out.finitemarginal, params = m_params)
end