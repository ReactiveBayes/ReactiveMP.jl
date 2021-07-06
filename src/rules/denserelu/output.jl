export rule

@rule DenseReLU(:output, Marginalisation) (q_input::MultivariateNormalDistributionsFamily, q_w::NTuple{N, MultivariateNormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where { N } = begin
    
    # assert whether the dimensions are correct
    @assert sum(length(q_input) .!= length.(q_w)) == 0 """
        The dimensionality of the input vector does not correspond to the dimensionality of the random variables representing the weights.

        The input variable x of dimensionality $(length(q_input)) looks like
        $(q_input)  
        whereas the first random variables of the weights of dimensionality $(length(q_w[1])) looks like
        $(q_w[1]).
    """

    # extract required statistics
    pz = mean.(q_z)
    mf = mean.(q_f)

    # extract parameters
    γ = getγ(meta)
    
    # calculate new statistics
    my = pz .* mf
    wy = diagm(ones(N)*γ)

    # return message
    return MvNormalMeanPrecision([my...], wy)

end