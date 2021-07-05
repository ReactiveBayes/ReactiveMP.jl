export rule

@rule DenseReLU(:input, Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_w::NTuple{N, MultivariateNormalDistributionsFamily}, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) where { N } = begin
    
    # extract required statistics
    mw, vw = unzip(mean_cov.(q_w))
    mf = mean.(q_f)

    # extract parameters
    β = getβ(meta)
    
    # calculate new statistics
    dim = length(mw)
    tmp = zeros(dim, dim)
    tmp2 = zeros(dim)
    for k = 1:dim
        @inbounds tmp += mw[k] * mw[k]' + vw[k]
        @inbounds tmp2 += mf[k]*mw[k]
    end
    wf = β .* tmp
    mf = cholinv(tmp) * tmp2

    # return message
    return MvNormalMeanPrecision(mf, wf)

end

# helper for broadcasting with multiple return values
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))