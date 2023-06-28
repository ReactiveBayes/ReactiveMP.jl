
@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::PointMass, meta::Union{<:AbstractCorrection, Nothing}) = PointMass(mean(m_A) \ mean(m_out))

@rule typeof(*)(:in, Marginalisation) (m_out::GammaDistributionsFamily, m_A::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return GammaShapeRate(shape(m_out), rate(m_out) * mean(m_A))
end

# if A is a matrix, then the result is multivariate
@rule typeof(*)(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_A::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A' * W_out * A)
    return MvNormalWeightedMeanPrecision(A' * ξ_out, W)
end

# if A is a vector, then the result is univariate
# this rule links to the special case (AbstractVector * Univariate) for forward (:out) rule
@rule typeof(*)(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_A::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, dot(A, W_out, A))
    return NormalWeightedMeanPrecision(dot(A, ξ_out), W)
end

# if A is a scalar, then the input is either univariate or multivariate
@rule typeof(*)(:in, Marginalisation) (m_out::F, m_A::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) where {F <: NormalDistributionsFamily} = begin
    A = mean(m_A)
    @logscale -logdet(A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A^2 * W_out)
    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), A * ξ_out, W)
end

# specialized versions for mean-covariance parameterization
@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_A)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, tmp * A)
    @logscale log(1 / sqrt(det(W) * det(Σ_out))) + getlogscale(messages[1]) #correct 

    return MvNormalWeightedMeanPrecision(tmp * μ_out, W)
end

@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_A)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, dot(tmp, A))

    return NormalWeightedMeanPrecision(dot(tmp, μ_out), W)
end


### Test for gp
@rule typeof(*)(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_A::UnivariateGaussianDistributionsFamily, meta::Tuple{ProcessMeta, TinyCorrection}) = begin
    return @call_rule typeof(*)(:A, Marginalisation) (m_out = m_out, m_in = m_A, meta = meta)
end

@rule typeof(*)(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_A::LogNormal, meta::Tuple{ProcessMeta, TinyCorrection}) = begin
    return @call_rule typeof(*)(:A, Marginalisation) (m_out = m_out, m_in = m_A, meta = TinyCorrection())
end 

@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::UnivariateGaussianDistributionsFamily, meta::TinyCorrection) = begin 
    backward_in = (x) -> -log(abs(x)) + logpdf(m_A,mean(m_out)/x)
    scalefactor = (x) -> exp(backward_in(x))/exp(-x^2)
    points, w = ReactiveMP.gausshermite(9)
    Z = dot(w,scalefactor.(points))
    # nsamples = 3000
    # samples = rand(StableRNG(1),m_A,nsamples)
    # Z = sum(1 ./ abs.(samples)) / nsamples
    if isnothing(messages[2].addons)
        @logscale log(Z)
    else
        @logscale log(Z) + getlogscale(messages[2]) #correct 
    end
    return ContinuousUnivariateLogPdf(backward_in)
end


@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::GaussianProcess, meta::ProcessMeta) = begin 
    index = meta.index
    m_gp, cov_gp = mean_cov(m_A.finitemarginal)
    μ_in = m_gp[index]
    var_in = cov_gp[index,index]
    return @call_rule typeof(*)(:in, Marginalisation) (m_out=m_out,m_A=NormalMeanVariance(μ_in,var_in),meta=TinyCorrection())
end

@rule typeof(*)(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_A::UnivariateGaussianDistributionsFamily, meta::TinyCorrection) = begin 
    μ_in, var_in = mean_var(m_A)
    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * abs(var_in + var_out / x^2))  - 1/2 * (μ_out / x - μ_in)^2 / (var_in + var_out / x^2)
    return ContinuousUnivariateLogPdf(backwardpass)
end



@rule typeof(*)(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_A::GaussianProcess, meta::ProcessMeta) = begin 
    index = meta.index
    m_gp, cov_gp = mean_cov(m_A.finitemarginal)
    d_A = NormalMeanVariance(m_gp[index], cov_gp[index,index])
    return @call_rule typeof(*)(:in, Marginalisation) (m_out=m_out,m_A=d_A,meta=TinyCorrection())
end



# @rule typeof(*)(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_A::UnivariateGaussianDistributionsFamily, meta::Union{<:AbstractCorrection, Nothing}) = begin
#     μ_A, var_A = mean_var(m_A)
#     μ_out, var_out = mean_var(m_out)
#     log_backwardpass = (x) -> -log(abs(x)) - 0.5 * log(2π * (var_A + var_out / x^2)) - 1 / 2 * (μ_out - x * μ_A)^2 / (var_A * x^2 + var_out)
#     return ContinuousUnivariateLogPdf(log_backwardpass)
# end

# @rule typeof(*)(:in, Marginalisation) (m_out::UnivariateDistribution, m_A::UnivariateDistribution, meta::Union{<:AbstractCorrection, Nothing}) = begin
#     return @call_rule typeof(*)(:A, Marginalisation) (m_out = m_out, m_in = m_A, meta = meta)
# end
