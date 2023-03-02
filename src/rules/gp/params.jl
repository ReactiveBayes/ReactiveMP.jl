## here we use CVI to estimate the marginal of `params` edge
import KernelFunctions: kernelmatrix, Kernel 
import LinearAlgebra: det, logdet 
import ToeplitzMatrices: Circulant
#----------- CVI ----------#
# Univariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, 
        q_params::UnivariateGaussianDistributionsFamily, meta::CVI) = begin 
    #collect entities in meta
    n_iter = meta.n_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc
    kernelfunc = q_kernelfunc  # this is already a function k(θ)
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

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, 
        m_params::UnivariateGaussianDistributionsFamily, meta::CVI) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = q_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end


# Multivariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, 
        q_params::MultivariateGaussianDistributionsFamily, meta::CVI) = begin 
    #collect entities in meta
    n_iter = meta.n_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc
    kernelfunc = q_kernelfunc  # this is already a function k(θ)
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

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, m_params::MultivariateGaussianDistributionsFamily, meta::CVI) = begin
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out=q_out, q_meanfunc=q_meanfunc, q_kernelfunc=q_kernelfunc, q_params=m_params, meta=meta)
end

@rule GaussianProcess(:params, Marginalisation) (m_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, meta::CVI) = begin 
    return NormalMeanVariance(0., 1.)
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
### (incorrect update)
###### Univariate case ###########
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::UnivariateGaussianDistributionsFamily, meta::Unscented) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc
    kernelfunc = q_kernelfunc  # this is already a function k(θ)

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

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, m_params::NormalMeanVariance, meta::Unscented{Int64, Float64, Int64, Nothing}) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = q_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end
# function prod(meta::Unscented, left::ContinuousUnivariateLogPdf, right::UnivariateGaussianDistributionsFamily)
#     Z,_ = approximate(meta, z -> exp(left.logpdf(z)), (mean(right),), (var(right),))
#     m, _ = approximate(meta, z -> exp(left.logpdf(z))*z/Z, (mean(right),), (var(right),))
#     v, _ = approximate(meta, z -> exp(left.logpdf(z))*(z-m)*(z-m)/Z, (mean(right),), (var(right),))

#     return NormalMeanVariance(m,v)
# end

### Multivariate Case #####
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::MultivariateGaussianDistributionsFamily, meta::Unscented) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc
    kernelfunc = q_kernelfunc  # this is already a function k(θ)
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

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, m_params::MvNormalMeanCovariance, meta::Unscented{Int64, Float64, Int64, Nothing}) = begin 
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
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::UnivariateGaussianDistributionsFamily, meta::GaussHermiteCubature) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    #new thing
    cov_strategy = q_out.covariance_strategy
    test = q_out.testinput 
    meanf = q_meanfunc
    kernelfunc = q_kernelfunc  # this is already a function k(θ)
    #get information from observations 
    y, Σ = mean_cov(q_out.finitemarginal) 
    inducing = q_out.inducing_input
    # do GaussHermiteCubature 
    msg_in = q_params
    #use "inv" instead of "cholinv"
    g = (x) -> inv_cov_mat(cov_strategy, kernelfunc, test, Σ, x, inducing)
    logp_nc = (x) -> - 1/2 * (y - meanf.(test))' * g(x)*(y - meanf.(test)) + 1/2 * logdet(g(x)) - length(test)/2 * log(2π)  #maximize
    result = prod(meta, ContinuousUnivariateLogPdf(logp_nc), msg_in)
    return result
end

@rule GaussianProcess(:params, Marginalisation) (m_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, m_params::UnivariateGaussianDistributionsFamily, meta::GaussHermiteCubature) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = m_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params, meta = meta)
end

@rule GaussianProcess(:params, Marginalisation) (m_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::UnivariateGaussianDistributionsFamily, meta::GaussHermiteCubature) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = m_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params, meta = meta)
end

function prod(meta::GaussHermiteCubature, left::ContinuousUnivariateLogPdf, right::UnivariateGaussianDistributionsFamily)
    m,v = approximate_meancov(meta, z -> exp(left.logpdf(z)), mean(right), var(right))
    return NormalMeanVariance(m,v)
end

### Multivariate Case #####
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::MultivariateGaussianDistributionsFamily, meta::GaussHermiteCubature) = begin 
    #collect entities in meta
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc
    kernelfunc = q_kernelfunc  # this is already a function k(θ)
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

### PAD 
@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::UnivariateGaussianDistributionsFamily) = begin 
    mean_in, var_in = mean_var(q_params)
    y, _= mean_cov(q_out.finitemarginal)
    circular_y = circularize(y)
    test = q_out.testinput
    circular_test = circularize(test)
    #kernelfunc(θ) is the first column of the circulant covariance matrix 
    mean_difference = circular_y - q_meanfunc.(circular_test)

    logp_llh = (θ) -> - 1/2 * fastinvmahalanobis_modified(q_kernelfunc(θ),mean_difference)  - 1/2 * logdet(Circulant(q_kernelfunc(θ))) - 0.5*length(circular_test) * log(2π)

    nsamples = 1000
    samples = rand(q_params,nsamples)
    weights = exp.(logp_llh.(samples)) / sum(exp.(logp_llh.(samples)) )
    weights[1:5]
    if any(isnan.(weights)) 
        m_ = sum(samples)/nsamples
        v_ = sum((samples .- m_).^2) /nsamples 
    else
        m_ = sum(weights .* samples)
        v_ = sum(weights .* (samples .- m_).^2)    
    end
    ksi = m_/v_ - mean_in/var_in
    precision = clamp(1/v_ - 1/var_in, tiny,huge)
            
    return NormalWeightedMeanPrecision(ksi,precision)
end

@rule GaussianProcess(:params, Marginalisation) (m_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, m_params::UnivariateGaussianDistributionsFamily) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = m_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = m_params)
end

@rule GaussianProcess(:params, Marginalisation) (m_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::UnivariateGaussianDistributionsFamily) = begin 
    return @call_rule GaussianProcess(:params, Marginalisation) (q_out = m_out, q_meanfunc = q_meanfunc, q_kernelfunc = q_kernelfunc, q_params = q_params)
end
#------------- compute ep message -----------#
# function ep_message end 

# function ep_message(meta::GaussHermiteCubature, m_in::UnivariateGaussianDistributionsFamily, GP::GaussianProcess , meanfunc, kernelfunc; nsamples = 1000)
#     #write the pdf of gaussian w.r.t. params 
#     mean_in, var_in = mean_var(m_in)
#     y, _= mean_cov(GP.finitemarginal)
#     circular_y = circularize(y)
#     test = GP.testinput
#     circular_test = circularize(test)

#     #kernelfunc(θ) is the first column of the circulant covariance matrix 
#     mean_difference = circular_y - meanfunc.(circular_test)

#     logp_llh = (θ) -> - 1/2 * fastinvmahalanobis_Hoang(kernelfunc(θ),mean_difference)  - 1/2 * log(det(Circulant(kernelfunc(θ)))) - 0.5*length(test) * log(2π)


#     samples = rand(m_in,nsamples)
#     weights = exp.(logp_llh.(samples)) / sum(exp.(logp_llh.(samples)) )
#     if any(isnan.(weights)) 
#         m_ = sum(samples)/nsamples
#         v_ = sum((samples .- m_).^2) /nsamples 
#     else
#         m_ = sum(weights .* samples)
#         v_ = sum(weights .* (samples .- m_).^2)    
#     end
#     ksi = m_/v_ - mean_in/var_in
#     precision = clamp(1/v_ - 1/var_in, tiny,huge)
            
#     return NormalWeightedMeanPrecision(ksi,precision)
# end



#-------helper function--------#
function circularize(arr)
    arrlength      = length(arr)
    circularlength = 2*(arrlength-1)
    
    circulararr    = Array{eltype(arr)}(undef,circularlength)
    
    @views circulararr[1:arrlength] .= view(arr,:,1)
    @inbounds [circulararr[i] = arr[arrlength - i%arrlength] for i=arrlength+1:circularlength]
    
    return circulararr
end

function fastinvmahalanobis(a, b , Afftmatrix,afft,bfft)
    mul!(afft, Afftmatrix, a)
    mul!(bfft, Afftmatrix, b)

    return real((2*sum(abs.(bfft).^2 ./ afft) - (abs(bfft[1])^2/afft[1] + abs(bfft[length(afft)])^2/afft[length(afft)]))/length(a))
end
function returnFFTstructures(circularvector)
    N                   = length(circularvector)
    n                   = Int(N/2 + 1)
    Afftmatrix          = plan_rfft(circularvector)
    afft                = Vector{ComplexF64}(undef, n)
    bfft                = Vector{ComplexF64}(undef, n)
    cfft                = Vector{Float64}(undef, N)
    Ainvfftmatrix       = plan_irfft(afft, N)
    
    return Afftmatrix,Ainvfftmatrix,afft,bfft,cfft 
end

function fastinvmahalanobis_modified(a,b)
    circular_a = circularize(a)
    Afftmatrix,_,afft,bfft,_ = returnFFTstructures(circular_a)
    return fastinvmahalanobis(circular_a,b,Afftmatrix,afft,bfft)
end