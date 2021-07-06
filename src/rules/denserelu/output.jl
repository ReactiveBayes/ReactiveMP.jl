export rule

@rule DenseReLU(:output, Marginalisation) (q_input::NormalDistributionsFamily, q_w::NTuple{N, NormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where { N } = begin
    
    # check whether a bias term is included
    use_bias = getuse_bias(meta)

    # assert whether the dimensions are correct
    @assert sum(length(q_input) + 1*use_bias .!= length.(q_w)) == 0 """
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

@rule DenseReLU(:output, Marginalisation) (q_input::NormalDistributionsFamily, q_w::NTuple{1, NormalDistributionsFamily}, q_z::NTuple{1, Bernoulli}, q_f::NTuple{1, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) = begin
    
    # check whether a bias term is included
    use_bias = getuse_bias(meta)

    # assert whether the dimensions are correct
    @assert sum(length(q_input) + 1*use_bias .!= length.(q_w)) == 0 """
        The dimensionality of the input vector does not correspond to the dimensionality of the random variables representing the weights.

        The input variable x of dimensionality $(length(q_input)) looks like
        $(q_input)  
        whereas the first random variables of the weights of dimensionality $(length(q_w[1])) looks like
        $(q_w[1]).
    """

    # extract required statistics
    pz = mean.(q_z)[1]
    mf = mean.(q_f)[1]

    # extract parameters
    γ = getγ(meta)
    
    # calculate new statistics
    my = pz * mf
    wy = γ

    # return message
    return NormalMeanPrecision(my, wy)

end