export rule

@rule DenseReLU((:w, k), Marginalisation) (q_output::NormalDistributionsFamily, q_input::MultivariateNormalDistributionsFamily, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # no dimensionality test required, because of typing.

    # extract required statistics
    mf = mean(q_f)
    mx, vx = mean_cov(q_input)

    # extract parameters
    β = getβ(meta)

    # calculate new statistics
    tmp = mx*mx'+ vx
    mw = mf *cholinv(tmp)*mx
    ww = β*tmp

    # return message
    return MvNormalMeanPrecision(mw, ww)

end

@rule DenseReLU((:w, k), Marginalisation) (q_output::NormalDistributionsFamily, q_input::UnivariateNormalDistributionsFamily, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # no dimensionality test required, because of typing.

    # extract required statistics
    mf = mean(q_f)
    mx, vx = mean_cov(q_input)

    # extract parameters
    β = getβ(meta)

    # calculate new statistics
    tmp = mx^2 + vx
    mw = mf/tmp*mx
    ww = β*tmp

    # return message
    return NormalMeanPrecision(mw, ww)

end