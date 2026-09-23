@testmodule ToyRules begin
    using MessagePassingRulesBase, Distributions, BayesBase
    using MessagePassingRulesBase: BP, VMP, annotate!

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :σ])

    @define_message_update_rule(
        node = Gauss, towards = :out, args = (m[:μ]::PointMass, m[:σ]::PointMass),
        body = (args) -> Normal(mean(args.m[:μ]), mean(args.m[:σ])),
    )
    @define_message_update_rule(
        node = Gauss, towards = :μ, args = (m[:out]::Normal, m[:σ]::PointMass),
        body = (args, ann) -> begin
            annotate!(ann, :logscale, 0.0)
            Normal(mean(args.m[:out]), sqrt(var(args.m[:out]) + mean(args.m[:σ])^2))
        end,
    )
    @define_marginal_update_rule(
        node = Gauss, towards = (:out, :μ), args = (m[:out]::Normal, m[:μ]::Normal, q[:σ]::PointMass),
        body = (args) -> promote(mean(args.m[:out]), mean(args.m[:μ])) .* one(mean(args.q[:σ])),
    )
    @define_average_energy(
        node = Gauss, args = (q[:out]::Normal, q[:μ]::Normal, q[:σ]::PointMass),
        body = (args) -> (var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ]))) / (2 * mean(args.q[:σ])^2),
    )

    # Negative controls.
    struct Faulty end
    @define_factor_node(node = Faulty, type = Stochastic, interfaces = [:out, :x])
    # Always Float64, whatever the input type: fails type promotion.
    @define_message_update_rule(node = Faulty, towards = :out, args = (m[:x]::PointMass,), body = (args) -> PointMass(Float64(mean(args.m[:x]))))
    # Returns a Float32 where Float64 is expected: fails the type check.
    @define_message_update_rule(node = Faulty, towards = :x, args = (m[:out]::PointMass,), body = (args) -> PointMass(Float32(mean(args.m[:out]))))

    struct Buffered end
    @define_factor_node(node = Buffered, type = Deterministic, interfaces = [:out, :x])
    @define_message_update_rule(
        node = Buffered, towards = :out, inplace = true, args = (m[:x]::Vector{Float64},),
        preallocate = (args) -> similar(args.m[:x]),
        body = (output::Vector{Float64}, args) -> (output .= 2 .* args.m[:x]; output),
    )
    # Generic over the element type, so dual numbers pass through.
    struct Scaled end
    @define_factor_node(node = Scaled, type = Deterministic, interfaces = [:out, :x])
    @define_message_update_rule(
        node = Scaled, towards = :out, inplace = true, args = (m[:x]::AbstractVector,),
        preallocate = (args) -> similar(args.m[:x]),
        body = (output::AbstractVector, args) -> (output .= 2 .* args.m[:x]; output),
    )

    # `rule!` ignores its buffer and allocates: disagrees on identity and allocates.
    @define_message_update_rule(
        node = Buffered, towards = :x, inplace = true, args = (m[:out]::Vector{Float64},),
        preallocate = (args) -> similar(args.m[:out]),
        body = (output::Vector{Float64}, args) -> args.m[:out] ./ 2,
    )
end
