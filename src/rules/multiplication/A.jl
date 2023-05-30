import ForwardDiff: hessian
import StableRNGs: StableRNG
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
    μ_in = m_gp[index]
    var_in = cov_gp[index,index]
    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  - 1/2 * (μ_out  - x*μ_in)^2 / (var_in*x^2 + var_out)
    return ContinuousUnivariateLogPdf(backwardpass)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::Tuple{ProcessMeta, TinyCorrection}) = begin 
    μ_in, var_in = mean_var(m_in)
    μ_out, var_out = mean_var(m_out)
    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  - 1/2 * (μ_out  - x*μ_in)^2 / (var_in*x^2 + var_out)
    return ContinuousUnivariateLogPdf(backwardpass)
end


function make_productdist_message(samples_in,d_out)
    return let samples_in=samples_in,d_out=d_out
        (x) -> begin
            result = mapreduce(+, zip(samples_in,)) do (samplein,)
                return samplein * pdf(d_out,x*samplein) #Ismail code: abs(samplein * pdf(d_out, x*samplein))
            end
            return log(result)
        end
    end
end


@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::LogNormal, meta::TinyCorrection) = begin
    nsamples    = 3000
    samples_in = rand(m_in,nsamples)
    samples_out = rand(m_out,nsamples)
    #this rule is used 
    p = make_productdist_message(samples_in,m_out)
    return ContinuousUnivariateLogPdf(p)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::TinyCorrection) = begin
    μ_in, var_in = mean_var(m_in)
    μ_out, var_out = mean_var(m_out)

    backwardpass = (x) -> -log(abs(x)) - 0.5*log(2π * (var_in + var_out / x^2))  - 1/2 * (μ_out  - x*μ_in)^2 / (var_in*x^2 + var_out)

    return ContinuousUnivariateLogPdf(backwardpass)    
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::LogNormal, meta::ProcessMeta) = begin
    return @call_rule typeof(*)(:A, Marginalisation) (m_out=m_out,m_in=m_in,meta=TinyCorrection())
end

@rule typeof(*)(:A, Marginalisation) (m_out::PointMass, m_in::LogNormal, meta::ProcessMeta) = begin 
    m_out_proxy = GaussianMeanVariance(mean(m_out),0.0001)
    return @call_rule typeof(*)(:A,Marginalisation) (m_out=m_out_proxy, m_in=m_in,meta=meta)
end


@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::LogNormal, meta::ProcessMeta) = begin 
    return @call_rule typeof(*)(:A, Marginalisation) (m_out=m_out,m_in=m_in,meta=TinyCorrection())
end


@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::Union{<:AbstractCorrection, Nothing}) = begin
    μ_in, var_in = mean_var(m_in)
    μ_out, var_out = mean_var(m_out)
    log_backwardpass = (x) -> -log(abs(x)) - 0.5 * log(2π * (var_in + var_out / x^2)) - 1 / 2 * (μ_out - x * μ_in)^2 / (var_in * x^2 + var_out)
    return ContinuousUnivariateLogPdf(log_backwardpass)
end

@rule typeof(*)(:A, Marginalisation) (m_out::UnivariateDistribution, m_in::UnivariateDistribution, meta::Union{<:AbstractCorrection, Nothing}) = begin
    nsamples = 3000
    samples_in = rand(m_in, nsamples)
    p = make_inversedist_message(samples_in, m_out)
    return ContinuousUnivariateLogPdf(p)
end

function make_inversedist_message(samples_in, d_out)
    return let samples_in = samples_in, d_out = d_out
        (x) -> begin
            result = mapreduce(+, samples_in) do samplein
                return abs(samplein) * pdf(d_out, x * samplein)
            end
            return log(result)
        end
    end
end
