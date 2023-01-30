# This rule isn't Belief Propagation nor VMP. 
# The message toward the `out` edge is a Gaussian process 
# Remember that we input the log of params

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, q_params::Any) = begin 
    θ = exp.(mean(q_params)) # we take exponential since mean(q_params) returns log value of θ
    kernelfunc = q_kernelfunc(θ)
    # return GaussianProcess(q_meanfunc,kernelfunc,nothing,Float64[],Float64[],Float64[],Float64[1;;],Float64[], CovarianceMatrixStrategy())
    return GaussianProcess(q_meanfunc,kernelfunc,nothing,Float64[],Float64[],Float64[], CovarianceMatrixStrategy())
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, m_params::Any) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params)
end

# when we have CVI meta 
@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, 
                                    q_params::Any, meta::CVI) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, 
                                    m_params::Any, meta::CVI) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, 
                                    q_params::Any, meta::UT) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, q_params::NormalMeanVariance, meta::GaussHermiteCubature) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, q_params::NormalWeightedMeanPrecision, meta::GaussHermiteCubature{Vector{Float64}, Vector{Float64}}) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, q_params::MvNormalMeanCovariance, meta::GaussHermiteCubature{Vector{Float64}, Vector{Float64}}) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end

@rule GaussianProcess(:out, Marginalisation) (q_meanfunc::Any, q_kernelfunc::Any, q_params::MvNormalWeightedMeanPrecision, meta::GaussHermiteCubature{Vector{Float64}, Vector{Float64}}) = begin 
    return @call_rule GaussianProcess(:out, Marginalisation) (q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end
