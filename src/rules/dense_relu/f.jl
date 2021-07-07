export rule

@rule DenseReLU((:f, k), Marginalisation) (q_output::NormalDistributionsFamily, q_input::NormalDistributionsFamily, q_w::NormalDistributionsFamily, q_z::Bernoulli, meta::DenseReLUMeta) = begin
    
    # check whether a bias term is included
    use_bias = getusebias(meta)

    # assert whether the dimensions are correct
    @assert length(q_input) + 1*use_bias == length(q_w) """
        The dimensionality of the input vector does not correspond to the dimensionality of the weights.
        
        The input variable x of dimensionality $(length(q_input)) looks like
        $(q_input)  
        and the weigth variable w of dimensionality $(length(q_w)) looks like 
        $(q_w)
    """

    # extract required statistics
    pz = mean(q_z)
    mw = mean(q_w)
    mx = use_bias ? vcat(mean(q_input), 1) : mean(q_input)
    my = mean(q_output)[k]

    # extract parameters
    C = getC(meta)
    β = getβ(meta)
    γ = getγ(meta)
    ξ = getξ(meta)[k]
    
    # calculate new statistics
    wf = γ*pz + β + 2*C^2*(sigmoid(ξ) - 0.5)/2/ξ
    mf = 1/wf*(γ*pz*my + β*dot(mw,mx) + C*pz - C/2)
    setξk!(meta, k, sqrt(mf^2 + 1/wf))

    # return message
    return NormalMeanPrecision(mf, wf)

end