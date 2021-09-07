@marginalrule Probit(:out_in) (m_out::PointMass, m_in::UnivariateNormalDistributionsFamily, meta::ProbitMeta) = begin 

    # extract parameters
    mz, vz = mean_cov(m_in)
    p = mean(m_out)
    @assert p >= zero(p) && p <= one(p) "The Probit node only accepts messages on its output with values between 0 and 1."

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

    return ( out = m_out, in = NormalMeanVariance(mpz, vpz) )

end