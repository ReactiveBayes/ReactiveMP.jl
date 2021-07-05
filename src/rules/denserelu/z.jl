export rule

@rule DenseReLU((:z, k), Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_input::MultivariateNormalDistributionsFamily, q_w::MultivariateNormalDistributionsFamily, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    mf, vf = mean_cov(q_f)
    my = mean(q_output)[k]

    # extract parameters
    γ = getγ(meta)
    C = getC(meta)

    # calculate new statistics
    p = sigmoid(γ*my*mf + C*mf - γ/2*(mf^2 + vf))

    # return message
    return Bernoulli(p)

end
