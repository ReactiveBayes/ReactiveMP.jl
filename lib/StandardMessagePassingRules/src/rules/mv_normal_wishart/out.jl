# Known parameters only. ExponentialFamily's constructor keeps each parameter's own
# float type; a message has one.
function promoted_mv_normal_wishart(μ, W, λ, ν)
    T = promote_type(eltype(μ), eltype(W), typeof(λ), typeof(ν))
    return MvNormalWishart(convert(AbstractVector{T}, μ), convert(AbstractMatrix{T}, W), convert(T, λ), convert(T, ν))
end

@define_message_update_rule(
    node = MvNormalWishart, target = :out,
    args = (q[:μ]::PointMass, q[:W]::PointMass, q[:λ]::PointMass, q[:ν]::PointMass),
    logscale = 0,
    body = (args) -> promoted_mv_normal_wishart(mean(args.q[:μ]), mean(args.q[:W]), mean(args.q[:λ]), mean(args.q[:ν])),
)
