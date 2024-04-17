# @rule GeneralizedTProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any,q_degree::Any, q_params::Any) = begin 
#     θ = exp.(mean(q_params)) # we take exponential since mean(q_params) returns log value of θ
#     kernelfunc = q_kernelfunc(θ)
#     # return GeneralizedTProcess(q_meanfunc,kernelfunc,nothing,Float64[],Float64[],Float64[],Float64[1;;],Float64[], CovarianceMatrixStrategy())
#     return GeneralizedTProcess(q_meanfunc,kernelfunc,mean(q_degree),Float64[],Float64[],Float64[], Float64[], CovarianceMatrixStrategy())
# end

# @rule GeneralizedTProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any,m_degree::Any, m_params::Any) = begin 
#     return @call_rule GeneralizedTProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc,q_degree=m_degree, q_params = m_params)
# end

