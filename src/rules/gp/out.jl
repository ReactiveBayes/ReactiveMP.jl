# This rule isn't Belief Propagation nor VMP. 
# The message toward the `out` edge is a Gaussian process 
# Remember that we input the log of params

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass, q_params::Any) = begin 
    func = q_kernelfunc.point
    θ = exp.(mean(q_params)) # we take exponential since mean(q_params) returns log value of θ
    kernelfunc = func(θ)
    return GaussianProcess(q_meanfunc.point,kernelfunc,nothing,Float64[],Float64[],Float64[],Float64[1;;],Float64[], CovarianceMatrixStrategy())
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::Any) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params)
end

# when we have CVI meta 
@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass, 
                                    q_params::Any, meta::CVIApproximation) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::PointMass, q_kernelfunc::PointMass, 
                                    m_params::Any, meta::CVIApproximation) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end