## here we use CVI to estimate the marginal of `params` edge
import KernelFunctions: kernelmatrix, Kernel 
# Univariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        q_params::UnivariateGaussianDistributionsFamily, meta::CVIApproximation) = begin 
    #collect entities in meta
    n_iter = meta.num_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    train = q_out.traininput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    y = q_out.observation
    Σ = q_out.observation_noise
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    λ_init = naturalparams(msg_in)
    #use "inv" instead of "cholinv"
    logp_nc(x) = -1/2 * (y - meanf.(train))' * inv_cov_mat(cov_strategy,kernelfunc,train, Σ, x, inducing)*(y - meanf.(train)) + 1/2 * logdet(inv_cov_mat(cov_strategy,kernelfunc,train, Σ, x, inducing)) - length(train)/2 * log(2π)
    λ = renderCVI(logp_nc, n_iter, optimizer, RNG, λ_init, msg_in)
    return convert(NormalMeanVariance, λ)
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        m_params::UnivariateGaussianDistributionsFamily, meta::CVIApproximation) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = q_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end

# Multivariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        q_params::MultivariateGaussianDistributionsFamily, meta::CVIApproximation) = begin 
    #collect entities in meta
    n_iter = meta.num_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    train = q_out.traininput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)

    y = q_out.observation
    Σ = q_out.observation_noise
    #new thing
    cov_strategy = q_out.covariance_strategy
    inducing = q_out.inducing_input
    # do CVI 
    msg_in = q_params
    λ_init = naturalparams(msg_in)
    #use "inv" instead of "cholinv"
    logp_nc(x) = -1/2 * (y - meanf.(train))' * inv_cov_mat(cov_strategy,kernelfunc,train, Σ, x, inducing)*(y - meanf.(train)) + 1/2 * logdet(inv_cov_mat(cov_strategy,kernelfunc,train, Σ, x, inducing)) - length(train)/2 * log(2π)
    λ = renderCVI(logp_nc, n_iter, optimizer, RNG, λ_init, msg_in)
    return convert(MvNormalMeanCovariance, λ)
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::MultivariateGaussianDistributionsFamily, meta::CVIApproximation) = begin
    return @call_rule GaussianProcss(:params, Marginalisation) (q_out=q_out, q_meanfunc=q_meanfunc, q_kernelfunc=q_kernelfunc, q_params=m_params, meta=meta)
end


### function for estimating the covariance matrix w.r.t. chosen gp strategy 
function inv_cov_mat(::CovarianceMatrixStrategy{FullCovarianceStrategy}, kernfunc::Function, input::AbstractArray, Σ::AbstractArray, θ, inducing::Nothing)
    return inv(kernelmatrix(kernfunc(exp.(θ)),input,input) + Diagonal(Σ) + 1e-8*diageye(length(input)))
end

function inv_cov_mat(::CovarianceMatrixStrategy{DeterministicInducingConditional}, kernfunc::Function, input::AbstractArray, Σ::AbstractArray, θ, inducing::AbstractArray)
    Kuu = kernelmatrix(kernfunc(exp.(θ)),inducing,inducing) + 1e-5*diageye(length(inducing))
    Kfu = kernelmatrix(kernfunc(exp.(θ)),input,inducing)
    Λ = Diagonal(Σ)

    return inv(Λ) - inv(Λ) * Kfu * inv(Kuu + Kfu' *inv(Λ)*Kfu) * Kfu' * inv(Λ) + 1e-5*diageye(length(input))
end

function inv_cov_mat(::CovarianceMatrixStrategy{DeterministicTrainingConditional}, kernfunc::Function, input::AbstractArray, Σ::AbstractArray, θ, inducing::AbstractArray)
    Kuu = kernelmatrix(kernfunc(exp.(θ)),inducing,inducing) + 1e-5*diageye(length(inducing))
    Kfu = kernelmatrix(kernfunc(exp.(θ)),input,inducing)
    Λ = Diagonal(Σ)

    return inv(Λ) - inv(Λ) * Kfu * inv(Kuu + Kfu' *inv(Λ)*Kfu) * Kfu' * inv(Λ) +  1e-5*diageye(length(input))
end

function inv_cov_mat(::CovarianceMatrixStrategy{FullyIndependentTrainingConditional}, kernfunc::Function, input::AbstractArray, Σ::AbstractArray, θ, inducing::AbstractArray)
    Kuu = kernelmatrix(kernfunc(exp.(θ)),inducing,inducing) + 1e-5*diageye(length(inducing))
    Kfu = kernelmatrix(kernfunc(exp.(θ)),input,inducing)
    Kff = kernelmatrix(kernfunc(exp.(θ)),input, input) + 1e-5*diageye(length(input))
    Λ =  Diagonal(Kff - Kfu * inv(Kuu) * Kfu' + Σ)

    return inv(Λ) - inv(Λ) * Kfu * inv(Kuu + Kfu' *inv(Λ)*Kfu) * Kfu' * inv(Λ) + 1e-5*diageye(length(input))
end
