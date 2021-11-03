
@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::NTuple{N,  UnivariateNormalDistributionsFamily}, q_p::NTuple{N, GammaDistributionsFamily }) where { N } = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::NTuple{N,  MultivariateNormalDistributionsFamily}, q_p::NTuple{N, Wishart }) where { N } = begin
    πs = probvec(q_switch)
    d  = ndims(first(q_m))
    w  = mapreduce(x -> x[1] * mean(x[2]), +, zip(πs, q_p))
    xi = mapreduce(x -> x[1] * mean(x[2]) * mean(x[3]), +, zip(πs, q_p, q_m))
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::NTuple{N,  PointMass{T}}, q_p::NTuple{N, PointMass{T} }) where { N, T <: Real } = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::NTuple{N,  PointMass}, q_p::NTuple{N, PointMass }) where { N } = begin
    πs = probvec(q_switch)
    d  = ndims(first(q_m))
    w  = mapreduce(x -> x[1] * mean(x[2]), +, zip(πs, q_p))
    xi = mapreduce(x -> x[1] * mean(x[2]) * mean(x[3]), +, zip(πs, q_p, q_m))
    return MvNormalWeightedMeanPrecision(xi, w)
end