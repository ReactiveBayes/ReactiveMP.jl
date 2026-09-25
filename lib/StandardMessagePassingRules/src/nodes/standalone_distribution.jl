"""
    StandaloneDistribution

The node `out ~ d` for a distribution value `d`, of any family, given as the constant
`distribution`: what a model writes as `x ~ prior`, `prior` a distribution rather than a
family with its parameters. Its message towards `out` is `d` itself, and its average energy
the cross entropy `E_q[-log d(out)]`, `KL(q ‖ d) + H(q)`. v6's StandaloneDistributionNode.
"""
struct StandaloneDistribution end

@define_factor_node(node = StandaloneDistribution, type = Stochastic, interfaces = [:out, :distribution])

# ⟨-log d(out)⟩_q, as KL(q ‖ d) + H(q), which Distributions has for any pair it can compare.
@define_average_energy(
    node = StandaloneDistribution,
    args = (q[:out]::Any, q[:distribution]::PointMass{<:Distribution}),
    body = (args) -> kldivergence(args.q[:out], BayesBase.getpointmass(args.q[:distribution])) + entropy(args.q[:out]),
)
