export rule

@rule DenseReLU((:w, k), Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_input::MultivariateNormalDistributionsFamily, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    mf = mean(q_f)
    mx, vx = meancov(q_input)

    # extract parameters
    β = getβ(meta)

    # calculate new statistics
    tmp = mx*mx'+ Vx
    mw = mf * mx'*cholinv(tmp)
    ww = β*tmp

    # return message
    return MvNormalMeanPrecision(mw, ww)

end