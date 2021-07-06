export rule

@rule DenseReLU((:w, k), Marginalisation) (q_output::NormalDistributionsFamily, q_input::MultivariateNormalDistributionsFamily, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # no dimensionality test required, because of typing.

    # check whether a bias term is included
    use_bias = getuse_bias(meta)

    # extract required statistics
    mf = mean(q_f)
    mx, vx = mean_cov(q_input)

    # augment if bias is desired
    if use_bias
        mx = vcat(mx,1)
        vx = hcat(vcat(vx, zeros(1,length(mx)-1)), zeros(length(mx)))
        vx[end, end] = tiny
    end

    # extract parameters
    β = getβ(meta)

    # calculate new statistics
    tmp = mx*mx' + vx
    mw = mf *cholinv(tmp)*mx
    ww = β*tmp

    # return message
    return MvNormalMeanPrecision(mw, ww)

end

@rule DenseReLU((:w, k), Marginalisation) (q_output::NormalDistributionsFamily, q_input::UnivariateNormalDistributionsFamily, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) = begin
    
    # no dimensionality test required, because of typing.

    # check whether a bias term is included
    use_bias = getuse_bias(meta)

    # extract required statistics
    mf = mean(q_f)
    mx, vx = mean_cov(q_input)

    if use_bias
        mx = vcat(mx,1)
        vx = hcat(vcat(vx, zeros(1,length(mx)-1)), zeros(length(mx)))
        vx[end, end] = tiny
    end

    # extract parameters
    β = getβ(meta)

    # calculate new statistics
    tmp = use_bias ? mx*mx'+ vx : mx^2 + vx
    mw = use_bias ? mf *cholinv(tmp)*mx : mf/tmp*mx
    ww = β*tmp

    # return message
    return use_bias ? MvNormalMeanPrecision(mw, ww) : NormalMeanPrecision(mw, ww)

end