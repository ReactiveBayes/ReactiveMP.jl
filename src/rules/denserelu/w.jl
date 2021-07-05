export rule

@rule DenseReLU((:w, k), Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_input::MultivariateNormalDistributionsFamily, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
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