export rule

@rule DenseReLU((:w, k), Marginalisation) (q_output::Union{NormalDistributionsFamily, PointMass}, q_input::Union{MultivariateNormalDistributionsFamily, PointMass{V}}, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) where { V <: AbstractVector }= begin
    
    # no dimensionality test required, because of typing.

    # check whether a bias term is included
    use_bias = getusebias(meta)

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

@rule DenseReLU((:w, k), Marginalisation) (q_output::Union{NormalDistributionsFamily, PointMass}, q_input::Union{UnivariateNormalDistributionsFamily, PointMass{T}}, q_z::Bernoulli, q_f::UnivariateNormalDistributionsFamily, meta::DenseReLUMeta) where { T <: Real }= begin
    
    # no dimensionality test required, because of typing.

    # check whether a bias term is included
    use_bias = getusebias(meta)

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