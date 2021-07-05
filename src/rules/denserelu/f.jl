export rule

@rule DenseReLU(:f, Marginalisation) (q_output::PointMass, q_input::PointMass, q_w::NormalMeanPrecision, q_z::Bernoulli, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    mz = mean(q_z)
    mw = mean(q_w)
    mx = mean(q_input)
    my = mean(q_output)

    # extract parameters
    C = getC(meta)
    β = getβ(meta)
    γ = getγ(meta)
    ξ = getξ(meta)
    
    # calculate new statistics
    wf = γ*mz + β + 2*C^2*(sigmoid(ξ) - 1)/2/ξ
    mf = 1/wf*(γ*mz*my + β*mw*mx + C*mz - C/2)
    setξ!(meta, sqrt(mf^2 + 1/wf))

    # return message
    return GaussianMeanPrecision(mf, wf)

end