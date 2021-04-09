
@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::NTuple{N,  UnivariateNormalDistributionsFamily}, q_p::NTuple{N, GammaDistributionsFamily }) where { N } = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end

@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::AbstractVector{UnivariateNormalDistributionsFamily}, q_p::AbstractVector{GammaDistributionsFamily }) where { N } = begin
    πs = probvec(q_switch)
    return NormalMeanPrecision(sum(πs .* mean.(q_m)), sum(πs .* mean.(q_p)))
end
