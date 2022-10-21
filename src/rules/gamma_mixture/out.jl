
@rule GammaMixture{N}(:out, Marginalisation) (
    q_switch::Any,
    q_a::ManyOf{N, Any},
    q_b::ManyOf{N, GammaDistributionsFamily}
) where {N} = begin
    πs = probvec(q_switch)
    return GammaShapeRate(sum(πs .* mean.(q_a)), sum(πs .* mean.(q_b)))
end
