
@rule Probit(:in, Marginalisation) (m_out::Union{PointMass, Bernoulli}, ) = begin
    
    # extract parameters
    p = mean(m_out)

    # calculate outward message
    f = (z) -> log(1 - p + (2*p - 1)*normcdf(z))

    # return message
    return ContinuousUnivariateLogPdf(f)

end

@rule Probit(:in, Marginalisation) (m_out::Union{PointMass, Bernoulli}, m_in::UnivariateNormalDistributionsFamily) = begin

    # extract parameters
    mz, vz = mean_cov(m_in)
    p = mean(m_out)
    @assert p >=0 && p <=1 "The Probit node only accepts messages on its output with values between 0 and 1."

    # calculate auxiliary variables
    γ = mz/sqrt(1+vz)

    # calculate moments of g
    mom1_g = normcdf(γ)*mz + vz*normpdf(γ)/sqrt(1+vz)
    mom2_g = 2*mz*mom1_g + (vz - mz^2)*normcdf(γ) + vz^2*γ*normpdf(γ)/(1+vz)

    # calculate moments of posterior
    mom0_pz = 1 - p + (2*p - 1)*normcdf(γ)
    mom1_pz = 1/mom0_pz*((1-p)*mz + (2*p-1)*mom1_g)
    mom2_pz = 1/mom0_pz*((1-p)*(vz + mz^2) + (2*p-1)*mom2_g)

    # calculate parameters of posterior
    mpz = mom1_pz
    vpz = mom2_pz - mom1_pz^2
    vpz = min(max(vpz, tiny), vz-tiny) # ensure variance of marginal is not larger than the variance of the cavity distribution.

    # calculate parameters of outgoing message
    vz_out = 1/(1/vpz - 1/vz)
    mz_out = vz_out*(mpz/vpz - mz/vz)

    # return message
    return NormalMeanVariance(mz_out, vz_out)

end