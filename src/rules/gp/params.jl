## here we use CVI to estimate the marginal of `params` edge
import KernelFunctions: kernelmatrix, Kernel 
import LinearAlgebra: det, logdet 
# Univariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        q_params::UnivariateGaussianDistributionsFamily, meta::CVI) = begin 
    #collect entities in meta
    n_iter = meta.n_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    #get information from observations 
    y, Σ = mean_cov(q_out.finitemarginal) 
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * logdet(g(x)) - length(test)/2 * log(2π)  #maximize
    result = prod(meta, ContinuousUnivariateLogPdf(logp_nc), msg_in)
    return result
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        m_params::UnivariateGaussianDistributionsFamily, meta::CVI) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = q_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end

# Multivariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        q_params::MultivariateGaussianDistributionsFamily, meta::CVI) = begin 
    #collect entities in meta
    n_iter = meta.n_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    #get information from observations 
    y, Σ = mean_cov(q_out.finitemarginal) 
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    dim = length(mean(q_params))
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * logdet(g(x)) - length(test)/2 * log(2π)  #maximize
    result = prod(meta, ContinuousMultivariateLogPdf(dim,logp_nc), msg_in)
    return result
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::MultivariateGaussianDistributionsFamily, meta::CVI) = begin
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out=q_out, q_meanfunc=q_meanfunc, q_kernelfunc=q_kernelfunc, q_params=m_params, meta=meta)
end

#---------------------------------------------------------------------------#
### function for estimating the covariance matrix w.r.t. chosen gp strategy 
function inv_cov_mat(::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernfunc, input, Σ, θ, inducing)
    return inv(kernelmatrix(kernfunc(exp.(θ)),input,input) + Diagonal(Σ) + 1e-5 * I)
end

function inv_cov_mat(::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernfunc, input, Σ, θ, inducing)
    Kuu = kernelmatrix(kernfunc(exp.(θ)),inducing,inducing) + 1e-5 * I
    Kfu = kernelmatrix(kernfunc(exp.(θ)),input,inducing)
    Λ = Diagonal(Σ)
    invΛ = inv(Λ)
    return invΛ  - invΛ * Kfu * inv(Kuu + Kfu' * invΛ * Kfu + 1e-5*I) * Kfu' * invΛ + 1e-5 * I
end

function inv_cov_mat(::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernfunc, input, Σ, θ, inducing)
    Kuu = kernelmatrix(kernfunc(exp.(θ)),inducing,inducing) + 1e-5 * I
    Kfu = kernelmatrix(kernfunc(exp.(θ)),input,inducing)
    Λ = Diagonal(Σ)
    invΛ = inv(Λ)
    return invΛ  - invΛ * Kfu * inv(Kuu + Kfu' * invΛ * Kfu + 1e-5*I) * Kfu' * invΛ + 1e-5 * I
end

function inv_cov_mat(::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernfunc, input, Σ, θ, inducing)
    Kuu = kernelmatrix(kernfunc(exp.(θ)),inducing,inducing) + 1e-5 * I
    Kfu = kernelmatrix(kernfunc(exp.(θ)),input,inducing)
    Kff = kernelmatrix(kernfunc(exp.(θ)),input, input) + 1e-5 * I

    Qff = Kfu * inv(Kuu) * Kfu' + 1e-5 * I
    Λ =  Diagonal(Kff - Qff + Σ)
    invΛ = inv(Λ)
    return invΛ  - invΛ * Kfu * inv(Kuu + Kfu' * invΛ * Kfu + 1e-5*I) * Kfu' * invΛ + 1e-5 * I
end

#-------------------Unscented Transform------------------------#
###### Univariate case ###########
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, q_params::UnivariateGaussianDistributionsFamily, meta::Unscented) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)

    y, Σ = mean_cov(q_out.finitemarginal) 

    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input

    msg_in = q_params
    m_x, var_x = mean_var(q_params)
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * logdet(g(x)) - length(test)/2 * log(2π)  #maximize

    # start again 
    Kyy = kernelmatrix(kernelfunc(exp(m_x)),test,test)
    Wx_fw = inv(var(q_params)) 
    Wy_tilde = cholinv(Kyy + Σ)
    ζy_tilde = Wy_tilde * (meanf.(test) - y)

    
    (sigma_points, _, weights_c) = sigma_points_weights(meta, m_x, var_x)
    g_sigma = logp_nc.(sigma_points)
    d = length(m_x)
    @inbounds C_fw = sum(weights_c[k+1] * (sigma_points[k+1] - m_x) * (g_sigma[k+1] .- meanf.(test))' for k in 0:(2d))

    Wx_tilde = Wx_fw * C_fw * Wy_tilde * C_fw' * Wx_fw
    ζx_tilde = Wx_fw * C_fw * ζy_tilde

    return NormalWeightedMeanPrecision(ζx_tilde, Wx_tilde)
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::NormalMeanVariance, meta::Unscented{Int64, Float64, Int64, Nothing}) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = q_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end
# function prod(meta::Unscented, left::ContinuousUnivariateLogPdf, right::UnivariateGaussianDistributionsFamily)
#     Z,_ = approximate(meta, z -> exp(left.logpdf(z)), (mean(right),), (var(right),))
#     m, _ = approximate(meta, z -> exp(left.logpdf(z))*z/Z, (mean(right),), (var(right),))
#     v, _ = approximate(meta, z -> exp(left.logpdf(z))*(z-m)*(z-m)/Z, (mean(right),), (var(right),))

#     return NormalMeanVariance(m,v)
# end

### Multivariate Case #####
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, q_params::MultivariateGaussianDistributionsFamily, meta::Unscented) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    #get information from observations 
    y, Σ = mean_cov(q_out.finitemarginal) 
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    m_x, cov_x = mean_cov(q_params)
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * chollogdet(g(x)) - length(test)/2 * log(2π)  #maximize
    
    # start again 
    Kyy = kernelmatrix(kernelfunc(exp.(m_x)),test,test)
    Wx_fw = inv(cov(q_params)) 
    Wy_tilde = cholinv(Kyy + Σ)
    ζy_tilde = Wy_tilde * (meanf.(test) - y)

    
    (sigma_points, _, weights_c) = sigma_points_weights(meta, m_x, cov_x)
    g_sigma = logp_nc.(sigma_points)
    d = length(m_x)
    C_fw = sum(weights_c[k+1] * (sigma_points[k+1] - m_x) * (g_sigma[k+1] .- meanf.(test))' for k in 0:(2d))

    Wx_tilde = Wx_fw * C_fw * Wy_tilde * C_fw' * Wx_fw
    ζx_tilde = Wx_fw * C_fw * ζy_tilde

    return MvNormalWeightedMeanPrecision(ζx_tilde, Wx_tilde)
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::MvNormalMeanCovariance, meta::Unscented{Int64, Float64, Int64, Nothing}) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = q_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end

# function prod(meta::Unscented, left::ContinuousMultivariateLogPdf, right::MultivariateGaussianDistributionsFamily)
# Z,_ = approximate(meta, z -> exp(left.logpdf(z)), (mean(right),), (cov(right),))
# m, _ = approximate(meta, z -> exp(left.logpdf(z))*z/Z, (mean(right),), (cov(right),))
# v, _ = approximate(meta, z -> exp(left.logpdf(z))*(z-m)*(z-m)'/Z, (mean(right),), (cov(right),))
# return MvNormalMeanCovariance(m,v)
# end


#------------ GaussHermiteCubature ----------------#
###### Univariate case ###########
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, q_params::UnivariateGaussianDistributionsFamily, meta::GaussHermiteCubature) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    #get information from observations 
    y, Σ = mean_cov(q_out.finitemarginal) 
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * logdet(g(x)) - length(test)/2 * log(2π)  #maximize
    result = prod(meta, ContinuousUnivariateLogPdf(logp_nc), msg_in)
    return result
end


function prod(meta::GaussHermiteCubature, left::ContinuousUnivariateLogPdf, right::UnivariateGaussianDistributionsFamily)
    m,v = approximate_meancov(meta, z -> exp(left.logpdf(z)), mean(right), var(right))
    return NormalMeanVariance(m,v)
end

### Multivariate Case #####
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, q_params::MultivariateGaussianDistributionsFamily, meta::GaussHermiteCubature) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    #get information from observations 
    y, Σ = mean_cov(q_out.finitemarginal) 
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    dim = length(mean(q_params))
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * logdet(g(x)) - length(test)/2 * log(2π)  #maximize
    result = prod(meta, ContinuousMultivariateLogPdf(dim, logp_nc), msg_in)
    return result
end


function prod(meta::GaussHermiteCubature, left::ContinuousMultivariateLogPdf, right::MultivariateGaussianDistributionsFamily)
    m,v = approximate_meancov(meta, z -> exp(left.logpdf(z)), mean(right), cov(right))
    return MvNormalMeanCovariance(m,v)
end