export rule

@rule DenseReLU(:input, Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_w::NTuple{N, MultivariateNormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where { N } = begin
    
    # assert whether the dimensions are correct
    @assert length(q_output) == length(q_w) """
        The dimensionality of the output vector does not correspond to the number of random variables representing the weights.

        The output variable y of dimensionality $(length(q_output)) looks like
        $(q_output)  
        whereas there are $(N) random variables for the weights.
    """
    @assert length(q_output) == length(q_f) """
        The dimensionality of the output vector does not correspond to the number of random variables representing the auxiliary variable f.

        The output variable y of dimensionality $(length(q_output)) looks like
        $(q_output)  
        whereas there are $(N) random variables for the auxiliary variable f.
    """
    @assert length(q_output) == length(q_z) """
        The dimensionality of the output vector does not correspond to the number of random variables representing the auxiliary variable z.

        The output variable y of dimensionality $(length(q_output)) looks like
        $(q_output)  
        whereas there are $(N) random variables for the auxiliary variable z.
    """

    # extract required statistics
    mw, vw = unzip(mean_cov.(q_w))
    mf = mean.(q_f)

    # extract parameters
    β = getβ(meta)
    
    # calculate new statistics
    dim_in = length(mw[1])
    tmp = zeros(dim_in, dim_in)
    tmp2 = zeros(dim_in)
    for k = 1:dim_in
        @inbounds tmp += mw[k] * mw[k]' + vw[k]
        @inbounds tmp2 += mf[k]*mw[k]
    end
    wf = β .* tmp
    mf = cholinv(tmp) * tmp2

    # return message
    return MvNormalMeanPrecision(mf, wf)

end

@rule DenseReLU(:input, Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_w::NTuple{N, UnivariateNormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where { N } = begin
    
    # assert whether the dimensions are correct
    @assert length(q_output) == length(q_w) """
        The dimensionality of the output vector does not correspond to the number of random variables representing the weights.

        The output variable y of dimensionality $(length(q_output)) looks like
        $(q_output)  
        whereas there are $(N) random variables for the weights.
    """
    @assert length(q_output) == length(q_f) """
        The dimensionality of the output vector does not correspond to the number of random variables representing the auxiliary variable f.

        The output variable y of dimensionality $(length(q_output)) looks like
        $(q_output)  
        whereas there are $(N) random variables for the auxiliary variable f.
    """
    @assert length(q_output) == length(q_z) """
        The dimensionality of the output vector does not correspond to the number of random variables representing the auxiliary variable z.

        The output variable y of dimensionality $(length(q_output)) looks like
        $(q_output)  
        whereas there are $(N) random variables for the auxiliary variable z.
    """

    # extract required statistics
    mw, vw = unzip(mean_cov.(q_w))
    mf = mean.(q_f)

    # extract parameters
    β = getβ(meta)
    
    # calculate new statistics
    dim_in = length(mw[1])
    tmp = 0.0
    tmp2 = 0.0
    for k = 1:dim_in
        @inbounds tmp += mw[k]^2 + vw[k]
        @inbounds tmp2 += mf[k]*mw[k]
    end
    wf = β .* tmp
    mf = cholinv(tmp) * tmp2

    # return message
    return NormalMeanPrecision(mf, wf)

end

# helper for broadcasting with multiple return values
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))