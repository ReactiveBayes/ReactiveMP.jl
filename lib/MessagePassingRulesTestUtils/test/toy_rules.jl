@testmodule ToyRules begin
    using MessagePassingRulesBase, Distributions, BayesBase
    using MessagePassingRulesBase: annotate!

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :σ])

    @define_message_update_rule(
        node = Gauss, target = :out, args = (m[:μ]::PointMass, m[:σ]::PointMass),
        body = (args) -> Normal(mean(args.m[:μ]), mean(args.m[:σ])),
    )
    @define_message_update_rule(
        node = Gauss, target = :μ, args = (m[:out]::Normal, m[:σ]::PointMass),
        body = (args, ann) -> begin
            annotate!(ann, :logscale, 0.0)
            Normal(mean(args.m[:out]), sqrt(var(args.m[:out]) + mean(args.m[:σ])^2))
        end,
    )
    @define_marginal_update_rule(
        node = Gauss, target = (:out, :μ), args = (m[:out]::Normal, m[:μ]::Normal, q[:σ]::PointMass),
        body = (args) -> promote(mean(args.m[:out]), mean(args.m[:μ])) .* one(mean(args.q[:σ])),
    )
    # A cluster that factorises: q(out, μ, σ) = q(out, μ) q(σ). Every block carries the
    # promoted float type of all the inputs, so the `(:σ,)` block cannot pass `m[:σ]` through
    # unchanged the way v6's `v = m_v` does.
    @define_marginal_update_rule(
        node = Gauss, target = (:out, :μ, :σ), args = (m[:out]::Normal, m[:μ]::Normal, m[:σ]::PointMass),
        body = (args) -> begin
            out, μ, σ = promote(mean(args.m[:out]), mean(args.m[:μ]), mean(args.m[:σ]))
            FactorizedCluster((:out, :μ) => PointMass([out, μ]), (:σ,) => PointMass(σ))
        end,
    )
    @define_average_energy(
        node = Gauss, args = (q[:out]::Normal, q[:μ]::Normal, q[:σ]::PointMass),
        body = (args) -> (var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ]))) / (2 * mean(args.q[:σ])^2),
    )

    # Negative controls.
    struct Faulty end
    @define_factor_node(node = Faulty, type = Stochastic, interfaces = [:out, :x])
    # Always Float64, whatever the input type: fails type promotion.
    @define_message_update_rule(node = Faulty, target = :out, args = (m[:x]::PointMass,), body = (args) -> PointMass(Float64(mean(args.m[:x]))))
    # Returns a Float32 where Float64 is expected: fails the type check.
    @define_message_update_rule(node = Faulty, target = :x, args = (m[:out]::PointMass,), body = (args) -> PointMass(Float32(mean(args.m[:out]))))

    struct Buffered end
    @define_factor_node(node = Buffered, type = Deterministic, interfaces = [:out, :x])
    @define_message_update_rule(
        node = Buffered, target = :out, inplace = true, args = (m[:x]::Vector{Float64},),
        preallocate = (args) -> similar(args.m[:x]),
        body = (output::Vector{Float64}, args) -> (output .= 2 .* args.m[:x]; output),
    )
    # Generic over the element type, so dual numbers pass through.
    struct Scaled end
    @define_factor_node(node = Scaled, type = Deterministic, interfaces = [:out, :x])
    @define_message_update_rule(
        node = Scaled, target = :out, inplace = true, args = (m[:x]::AbstractVector,),
        preallocate = (args) -> similar(args.m[:x]),
        body = (output::AbstractVector, args) -> (output .= 2 .* args.m[:x]; output),
    )

    # Scratch: one that writes before reading, one that accumulates into its scratch and so
    # depends on what an earlier call left there.
    struct Scratched end
    @define_factor_node(node = Scratched, type = Deterministic, interfaces = [:out, :x])
    @define_message_update_rule(
        node = Scratched, target = :out, args = (m[:x]::Vector{Float64},),
        scratch = (args) -> (work = similar(args.m[:x]),),
        body = (scratch, args) -> (scratch.work .= 2 .* args.m[:x]; PointMass(sum(scratch.work))),
    )
    @define_message_update_rule(
        node = Scratched, target = :x, args = (m[:out]::PointMass,),
        scratch = (args) -> (total = zeros(1),),
        body = (scratch, args) -> (scratch.total[1] += mean(args.m[:out]); PointMass(scratch.total[1])),
    )

    # A rule over whatever inputs the factorisation delivers, with one typed input: it sums them.
    struct Summed end
    @define_factor_node(node = Summed, type = Stochastic, interfaces = [:out, :w, :T...])
    @define_message_update_rule(
        node = Summed, target = :out, args = (default, q[:w]::PointMass),
        body = (args) -> PointMass(sum(p -> mean(last(p)), MessagePassingRulesBase.rule_inputs(Summed, args.q)) + sum(p -> mean(last(p)), MessagePassingRulesBase.rule_inputs(Summed, args.m); init = 0.0)),
    )

    # Default rules an extension inherits. Reached under `Extended`, they run with
    # `DefaultAlgorithm()` in their `algo` slot, so a helper typed `::DefaultAlgorithm` is
    # reachable from the body and from `preallocate`.
    struct Extended <: DefaultAlgorithmExtension end
    only_default(::DefaultAlgorithm, x) = x
    struct Inherited end
    @define_factor_node(node = Inherited, type = Deterministic, interfaces = [:out, :x])
    @define_message_update_rule(
        node = Inherited, target = :out, args = (m[:x]::PointMass,),
        body = (algo, args) -> PointMass(2 * only_default(algo, mean(args.m[:x]))),
    )
    @define_message_update_rule(
        node = Inherited, target = :x, inplace = true, args = (m[:out]::AbstractVector,),
        preallocate = (algo, args) -> similar(only_default(algo, args.m[:out])),
        body = (output::AbstractVector, algo, args) -> (output .= only_default(algo, args.m[:out]) ./ 2; output),
    )

    # `rule!` ignores its buffer and allocates: disagrees on identity and allocates.
    @define_message_update_rule(
        node = Buffered, target = :x, inplace = true, args = (m[:out]::Vector{Float64},),
        preallocate = (args) -> similar(args.m[:out]),
        body = (output::Vector{Float64}, args) -> args.m[:out] ./ 2,
    )
end
