export rule

@rule DenseReLU(:output, Marginalisation) (q_input::MultivariateNormalDistributionsFamily, q_w::NTuple{N, MultivariateNormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where { N } = begin
    
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