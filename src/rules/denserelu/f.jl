export rule

@rule DenseReLU((:f, k), Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_input::MultivariateNormalDistributionsFamily, q_w::MultivariateNormalDistributionsFamily, q_z::Bernoulli, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    pz = mean(q_z)
    mw = mean(q_w)
    mx = mean(q_input)
    my = mean(q_output)[k]

    # extract parameters
    C = getC(meta)
    β = getβ(meta)
    γ = getγ(meta)
    ξ = getξ(meta)[k]
    
    # calculate new statistics
    wf = γ*pz + β + 2*C^2*(sigmoid(ξ) - 1)/2/ξ
    mf = 1/wf*(γ*pz*my + β*dot(mw,mx) + C*pz - C/2)
    setξk!(meta, k, sqrt(mf^2 + 1/wf))

    # return message
    return NormalMeanPrecision(mf, wf)

end