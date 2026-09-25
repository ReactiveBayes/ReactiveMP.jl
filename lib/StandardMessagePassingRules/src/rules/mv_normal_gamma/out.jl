# ExponentialFamily's constructor keeps each parameter's own float type; a message has one.
function promoted_mv_normal_gamma(μ, Λ, α, β)
    T = promote_type(eltype(μ), eltype(Λ), typeof(α), typeof(β))
    return MvNormalGamma(convert(AbstractVector{T}, μ), convert(AbstractMatrix{T}, Λ), convert(T, α), convert(T, β))
end

@define_message_update_rule(
    node = MvNormalGamma, target = :out,
    args = (m[:μ]::PointMass, m[:Λ]::PointMass, m[:α]::PointMass, m[:β]::PointMass),
    body = (args) -> promoted_mv_normal_gamma(mean(args.m[:μ]), mean(args.m[:Λ]), mean(args.m[:α]), mean(args.m[:β])),
)

# The parameters enter log MvNormalGamma linearly, except through γ (out - μ)ᵀΛ(out - μ) / 2,
# whose expectation adds tr(E[Λ] Cov μ) / 2 to the rate.
@define_message_update_rule(
    node = MvNormalGamma, target = :out,
    args = (q[:μ]::Any, q[:Λ]::Any, q[:α]::Any, q[:β]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.q[:μ])
        Λ = mean(args.q[:Λ])
        promoted_mv_normal_gamma(μ, Λ, mean(args.q[:α]), mean(args.q[:β]) + tr(Λ * V) / 2)
    end,
)
