export rule

@rule DenseReLU((:z, k), Marginalisation) (q_output::NormalDistributionsFamily, q_input::NormalDistributionsFamily, q_w::NormalDistributionsFamily, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # assert whether the dimensions are correct
    @assert length(q_input) == length(q_w) """
        The dimensionality of the input vector does not correspond to the dimensionality of the weights.

        The input variable x of dimensionality $(length(q_input)) looks like
        $(q_input)  
        and the weigth variable w of dimensionality $(length(q_w)) looks like 
        $(q_w)
    """

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
