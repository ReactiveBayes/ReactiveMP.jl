
@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::ManyOf{N, UnivariateNormalDistributionsFamily}, q_p::ManyOf{N, GammaDistributionsFamily}) where {N} = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::ManyOf{N, MultivariateNormalDistributionsFamily}, q_p::ManyOf{N, Wishart}) where {N} = begin
    πs = probvec(q_switch)
    d  = ndims(first(q_m))

    # Better to preinitialize
    q_p_m = mean.(q_p)
    q_m_m = mean.(q_m)

    W = mapreduce(x -> x[1] * x[2], +, zip(πs, q_p_m))
    ξ = mapreduce(x -> x[1] * x[2] * x[3], +, zip(πs, q_p_m, q_m_m))

    return MvNormalWeightedMeanPrecision(ξ, W)
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::ManyOf{N, PointMass{T} where T <: Real}, q_p::ManyOf{N, PointMass{T} where T <: Real}) where {N} = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::ManyOf{N, PointMass{<:AbstractVector}}, q_p::ManyOf{N, PointMass{<:AbstractMatrix}}) where {N} = begin
    πs = probvec(q_switch)
    d  = ndims(first(q_m))

    w  = mapreduce(x -> x[1] * mean(x[2]), +, zip(πs, q_p))
    xi = mapreduce(x -> x[1] * mean(x[2]) * mean(x[3]), +, zip(πs, q_p, q_m))

    return MvNormalWeightedMeanPrecision(xi, w)
end
