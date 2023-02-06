import ForwardDiff: hessian

@rule typeof(*)(:A, Marginalisation) (m_out::PointMass, m_in::PointMass, meta::Union{<:AbstractCorrection, Nothing}) = PointMass(mean(m_in) \ mean(m_out))

@rule typeof(*)(:A, Marginalisation) (m_out::GammaDistributionsFamily, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return GammaShapeRate(shape(m_out), rate(m_out) * mean(m_in))
end

# if A is a matrix, then the result is multivariate
@rule typeof(*)(:A, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A' * W_out * A)
    return MvNormalWeightedMeanPrecision(A' * ξ_out, W)
end

# if A is a vector, then the result is univariate
# this rule links to the special case (AbstractVector * Univariate) for forward (:out) rule
@rule typeof(*)(:A, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, dot(A, W_out, A))
    return NormalWeightedMeanPrecision(dot(A, ξ_out), W)
end

# if A is a scalar, then the input is either univariate or multivariate
@rule typeof(*)(:A, Marginalisation) (m_out::F, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) where {F <: NormalDistributionsFamily} = begin
    A = mean(m_in)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A^2 * W_out)
    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), A * ξ_out, W)
end

# specialized versions for mean-covariance parameterization
@rule typeof(*)(:A, Marginalisation) (m_out::MvNormalMeanCovariance, m_in::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, tmp * A)

    return MvNormalWeightedMeanPrecision(tmp * μ_out, W)
end

@rule typeof(*)(:A, Marginalisation) (m_out::MvNormalMeanCovariance, m_in::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, dot(tmp, A))

    return NormalWeightedMeanPrecision(dot(tmp, μ_out), W)
end

##### test gp
@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::GaussianProcess, meta::Tuple{ProcessMeta, TinyCorrection}) = begin
    index = meta[1].index
    m_gp, cov_gp = mean_cov(m_in.finitemarginal)
    kernelf = m_in.kernelfunction
    meanf   = m_in.meanfunction
    test    = m_in.testinput
    train   = m_in.traininput
    cov_strategy = m_in.covariance_strategy
    x_u = m_in.inducing_input
    mμ, var_μ = ReactiveMP.predictMVN(cov_strategy, kernelf,meanf,test,[train[index]],m_gp, x_u) #this returns negative variance
    μ_in = mμ[1]
    var_in = var_μ[1]

    # get mean and variance from m_out
    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  -1/2 * (μ_out / x - μ_in)^2 / (var_in + var_out / x^2)
    return ContinuousUnivariateLogPdf(backwardpass)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::Tuple{ProcessMeta, TinyCorrection}) = begin
    μ_in, var_in = mean_var(m_in)
    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  - 1/2 * (μ_out / x - μ_in)^2 / (var_in + var_out / x^2)

    return ContinuousUnivariateLogPdf(backwardpass)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::LogNormal, meta::TinyCorrection) = begin
    #laplace approximation
    res = optimize(x -> -logpdf(m_in,x), -50,50)
    μ_in= res.minimizer[1]

    # # hessian(x -> -logpdf(m_in,x), μ_in_approx)
    dx = (x) -> ForwardDiff.derivative(y -> -logpdf(m_in,y),x)
    ddx = (x) -> ForwardDiff.derivative(dx,x)

    var_in = cholinv(ddx(μ_in))
    # @show μ_in_approx, var_in_approx
    # μ_in_approx = mean(m_in)
    # var_in_approx = var(m_in) + 1e-5
    μ_out, var_out = mean_var(m_out)

    # backwardpass = (x) -> -log(abs(x)  +1e-2) - 0.5*log(2π * (var_in + var_out / (1e-2 + x^2)))  - 1/2 * (μ_out / (1e-2 + x) - μ_in)^2 / (var_in + var_out / (1e-2 + x^2))
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / (x^2)))  - 1/2 * (μ_out / (x ) - μ_in)^2 / (var_in + var_out / (x^2))
    res = optimize(x -> -backwardpass(x),-2.,2.)
    m_  = res.minimizer[1]
    
    dx = (x) -> ForwardDiff.derivative(y -> -backwardpass(y),x)
    ddx = (x) -> ForwardDiff.derivative(dx,x)

    v_ = cholinv(ddx(m_))

    # backwardpass_unsafe = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in_approx + var_out / x^2))  - 1/2 * (μ_out / x - μ_in_approx)^2 / (var_in_approx + var_out / x^2)
    # function backwardpass_unsafe(x)
    #     if x >= -1e-2 && x <= 1e-2
    #         tmp = log(1/(sqrt(2π*var_out)) * exp(-μ_out^2/(2*var_out)))
    #         tmp < log(1e-4) ? dist = logpdf(NormalMeanVariance(1e-4,0.1), x) : dist = logpdf(NormalMeanVariance(1/(sqrt(2π*var_out)) * exp(-μ_out^2/(2*var_out)),0.01), x)
    #         return dist
    #     else
    #         return -log(abs(x)) - 0.5*log(2π * (var_in_approx + var_out / x^2))  -1/2 * (μ_out / x - μ_in_approx)^2 / (var_in_approx + var_out / x^2)
    #     end
    # end

    # backwardpass = backwardpass_unsafe

    # return ContinuousUnivariateLogPdf(backwardpass)
    return NormalMeanVariance(m_, v_)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::TinyCorrection) = begin
    μ_in, var_in = mean_var(m_in)
    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  - 1/2 * (μ_out / x - μ_in)^2 / (var_in + var_out / x^2)

    return ContinuousUnivariateLogPdf(backwardpass)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::Tuple, meta::TinyCorrection) = begin
    μ_in = mean(m_in[1])
    var_in = var(m_in[1])

    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  -1/2 * (μ_out / x - μ_in)^2 / (var_in + var_out / x^2)

    return ContinuousUnivariateLogPdf(backwardpass)
end



@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::LogNormal, meta::ProcessMeta) = begin
    return @call_rule typeof(*)(:A, Marginalisation) (m_out=m_out,m_in=m_in,meta=TinyCorrection())
end
