using StatsFuns: normcdf, normccdf, normlogcdf, normlogccdf, normlogpdf, normpdf, logsumexp

@rule Probit(:in, Marginalisation) (m_out::Union{PointMass, Bernoulli}, ) = begin
    
    # extract parameters
    p = mean(m_out)

    # calculate outward message
    f = (z) -> log(1 - p + (2*p - 1)*normcdf(z))

    # return message
    return ContinuousUnivariateLogPdf(f)

end

@rule Probit(:in, MomentMatching) (m_out::Union{PointMass, Bernoulli}, m_in::UnivariateNormalDistributionsFamily) = begin

    # extract parameters
    mz, vz = mean_cov(m_in)
    p = mean(m_out)
    @assert p >= zero(p) && p <= one(p) "The Probit node only accepts messages on its output with values between 0 and 1."

    T = promote_type(eltype(m_out), eltype(m_in))

    # calculate auxiliary variables
    γ = mz/sqrt(1+vz)

    # calculate moments of g
    if γ > 0 && p > 0.5
        log_mom0_pz = logsumexp([log(1-p), log(2*p-1) + normlogccdf(-γ)])
    elseif γ <= 0 && p > 0.5
        log_mom0_pz = logsumexp([log(1-p), log(2*p-1) + normlogcdf(γ)])
    elseif γ > 0 && p <= 0.5
        log_mom0_pz = logsumexp([log(1-p) + normlogcdf(-γ), log(p) + normlogcdf(γ)])
    else
        log_mom0_pz = logsumexp([log(1-p) + normlogccdf(γ), log(p) + normlogcdf(γ)])
    end
    mom1_pz = mz + (2*p-1)*exp(log(vz) + normlogpdf(γ) - 0.5*log(1+vz) - log_mom0_pz)
    mom2_pz = vz + mz^2 + (2*p-1)*2*mz*exp(log(vz) + normlogpdf(γ) - 0.5*log(1+vz) - log_mom0_pz) - (2p-1)*γ*exp(2*log(vz) + normlogpdf(γ) - log(1 + vz) - log_mom0_pz)

    # calculate parameters of posterior
    mpz = mom1_pz
    vpz = mom2_pz - mom1_pz^2
    vpz = min(max(vpz, tiny), vz - tiny) # ensure variance of marginal is not larger than the variance of the cavity distribution.

    # calculate parameters of outgoing message
    wz_out = 1/vpz - 1/vz
    ξz_out = mpz/vpz - mz/vz

    # return message
    return NormalWeightedMeanPrecision{T}(ξz_out, wz_out)

end