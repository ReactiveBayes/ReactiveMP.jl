export rule

@rule DenseReLU(:input, Marginalisation) (q_output::MultivariateNormalDistributionsFamily, q_w::MultivariateNormalDistributionsFamily, q_z::NTuple{N, Bernoulli}, q_f::NTuple{N, UnivariateNormalDistributionsFamily}, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    mw, vw = meancov.(q_w)
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