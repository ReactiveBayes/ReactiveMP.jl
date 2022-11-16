## here we use CVI to estimate the marginal of `params` edge

# Univariate case 

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, 
        q_params::UnivariateGaussianDistributionsFamily, meta::CVIApproximation) = begin 
    #collect entities in meta
    n_iter = meta.num_iterations; 
    num_sample = meta.n_samples ;
    optimizer = meta.opt;
    RNG = meta.rng 
    #collect information from gaussian process q_out 
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    # kernfunc(x) = kernelfunc_userdefined(exp(x))
    y, Σ = mean_cov(q_out.finitemarginal)

    # do CVI 
    msg_in = q_params
    λ_init = naturalparams(msg_in)
    #use "inv" instead of "cholinv"
    logp_nc(x) = -1/2 * (y - meanf.(test))' * inv(kernelmatrix(kernfunc(exp(x)),test,test) + Diagonal(Σ) + 1e-8*diageye(length(test))) * (y- meanf.(test)) - 1/2 * logdet(kernelmatrix(kernfunc(exp(x)),test,test) + Diagonal(Σ) + 1e-8*diageye(length(test)))
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
    test = q_out.testinput 
    meanf = q_meanfunc.point
    kernelfunc = q_kernelfunc.point  # this is already a function k(θ)
    # kernfunc(x) = kernelfunc_userdefined(exp.(x))
    y, Σ = mean_cov(q_out.finitemarginal)

    # do CVI 
    msg_in = q_params
    λ_init = naturalparams(msg_in)
    #use "inv" instead of "cholinv"
    logp_nc(x) = -1/2 * (y - meanf.(test))' * inv(kernelmatrix(kernfunc(exp.(x)),test,test) + Diagonal(Σ) + 1e-8*diageye(length(test))) * (y- meanf.(test)) - 1/2 * logdet(kernelmatrix(kernfunc(exp.(x)),test,test) + Diagonal(Σ) + 1e-8*diageye(length(test)))
    λ = renderCVI(logp_nc, n_iter, optimizer, RNG, λ_init, msg_in)
    return convert(MvNormalMeanCovariance, λ)
end

@rule GaussianProcess(:params, Marginalisation) (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, m_params::MultivariateGaussianDistributionsFamily, meta::CVIApproximation) = begin
    return @call_rule GaussianProcss(:params, Marginalisation) (q_out=q_out, q_meanfunc=q_meanfunc, q_kernelfunc=q_kernelfunc, q_params=m_params, meta=meta)
end