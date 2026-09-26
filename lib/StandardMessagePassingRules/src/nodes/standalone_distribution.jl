"""
    StandaloneDistribution

The node `out ~ d` for a distribution value `d` of any family, given on its interface
`distribution` as a constant: what a model writes as `x ~ prior`, with `prior` a distribution
rather than a family and its parameters. Its interfaces are `out` and `distribution`, and it
runs under [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm).

**Rules.** The message towards `out` is `d` itself, from the marginal of `distribution`, a
`PointMass` holding `d`; its log scale is zero. There is no rule towards `distribution`, which
must be a constant. The average energy is the cross entropy
`E_q[-log d(out)] = KL(q ‖ d) + H(q)`, defined for every pair `(q, d)` for which Distributions
has a `kldivergence`.

# Examples

```jldoctest; setup = :(using StandardMessagePassingRules, MessagePassingRulesBase, BayesBase, Distributions)
julia> prior = Beta(2.0, 3.0);

julia> getresult(@call_message_update_rule(node = StandaloneDistribution, target = :out, q = (distribution = PointMass(prior),))) === prior
true
```

See also [`Uninformative`](@ref).
"""
struct StandaloneDistribution end

@define_factor_node(node = StandaloneDistribution, type = Stochastic, interfaces = [:out, :distribution])

# ⟨-log d(out)⟩_q, as KL(q ‖ d) + H(q), which Distributions has for any pair it can compare.
@define_average_energy(
    node = StandaloneDistribution,
    args = (q[:out]::Any, q[:distribution]::PointMass{<:Distribution}),
    body = (args) -> kldivergence(args.q[:out], BayesBase.getpointmass(args.q[:distribution])) + entropy(args.q[:out]),
)
