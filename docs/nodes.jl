# The node the examples of ReactiveMP's documentation run on. The engine defines no node: rule
# packages do, StandardMessagePassingRules among them. So that this site depends on none of them,
# its examples declare this small node with MessagePassingRulesBase, as a rule package would.
# Every page includes this file; the page "The example node" shows it.

using MessagePassingRulesBase, BayesBase, ExponentialFamily

# A normal density with mean `μ` and a known variance `v`: f(out, μ, v) = N(out | μ, v).
struct Gaussian end

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])

# The messages the rules take: a normal distribution, or a point mass (an observation or a
# constant).
const NormalOrPoint = Union{PointMass, UnivariateNormalDistributionsFamily}

## Belief propagation: the node is one cluster, and its rules take messages.

# N(out | m, s + v): the mean is integrated out against its message.
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (m[:μ]::NormalOrPoint, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)

# The same integral the other way: the density is symmetric in `out` and `μ`.
@define_message_update_rule(
    node = Gaussian, target = :μ,
    args = (m[:out]::NormalOrPoint, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), var(args.m[:out]) + mean(args.m[:v])),
)

# The joint marginal of the node's cluster, which the free energy needs. With one edge a point
# mass, it factorises: the point mass, and the product of the other edge's message with the
# node's density at the point.
@define_marginal_update_rule(
    node = Gaussian, target = (:out, :μ, :v),
    args = (m[:out]::PointMass, m[:μ]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
    body = (args) -> FactorizedCluster(
        (:out,) => args.m[:out],
        (:μ,) => prod(ClosedProd(), args.m[:μ], NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v]))),
        (:v,) => args.m[:v],
    ),
)

@define_marginal_update_rule(
    node = Gaussian, target = (:out, :μ, :v),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::PointMass, m[:v]::PointMass),
    body = (args) -> FactorizedCluster(
        (:out,) => prod(ClosedProd(), args.m[:out], NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v]))),
        (:μ,) => args.m[:μ],
        (:v,) => args.m[:v],
    ),
)

## Variational message passing: every edge a cluster of its own, and the rules take marginals.

# exp E[log N(out | μ, v)] ∝ N(out | E[μ], v).
@define_message_update_rule(
    node = Gaussian, target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)

@define_message_update_rule(
    node = Gaussian, target = :μ,
    args = (q[:out]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:out]), mean(args.q[:v])),
)

## The average energy, E[-log N(out | μ, v)], under independent marginals.

@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> begin
        out, μ, v = args.q[:out], args.q[:μ], mean(args.q[:v])
        (log(2v * π) + (var(out) + var(μ) + abs2(mean(out) - mean(μ))) / v) / 2
    end,
)
