@marginalrule GeneralizedTProcess(:out_params_degree) (m_out::GeneralizedTProcess, q_meanfunc::Any, q_kernelfunc::Any,m_degree::Any, m_params::Any) = begin
    return (out = m_out.finitemarginal, params = m_params,degree=m_degree)
end