# [Migrating from v6 to v7](@id migration-v6-to-v7)

ReactiveMP v7 moves nodes and rules out of the engine into packages of their own. The engine
keeps variables, factor nodes, messages, marginals and the free energy; `MessagePassingRulesBase`
defines how nodes and rules are declared and found; `StandardMessagePassingRules` holds the
standard nodes' rules; `MessagePassingRulesApproximations` and `DeltaMessagePassingRules` hold the
approximation methods and the Delta node. Most of a port is mechanical, and this guide lists the
mechanical translations as before/after pairs. The v7 side of each pair runs when these docs are
built.

## For an agent porting code

Read this section before changing anything.

- **Read first:** [Defining nodes and rules](@ref rules-defining) and
  [Algorithms and dependencies](@ref rules-algorithms), then the pairs below.
- **Translate only what a pair covers.** Each construct in the code you port must match a pair
  in this guide. Anything else, and everything in [What cannot be translated
  mechanically](@ref migration-v6-to-v7-manual), is a question for the person who owns the code.
  Stop and ask; do not guess. A rule that compiles and returns a plausible distribution can
  still be wrong, and nothing downstream will catch it.
- **Never guess** which inputs a rule consumes, whether an input is a message or a marginal, the
  order of interfaces, or what `meta` held: all of these change the result.
- **Verify every rule you port** as [Verifying a port](@ref migration-v6-to-v7-verify) describes:
  a table of cases, and a comparison with the old rule while it is available.
- **Stop** when a rule reads raw message tuples, builds graph objects, keeps state in `meta`, or
  when a verification fails and you cannot explain the difference.

## Nodes

A node is declared with `@define_factor_node`, keyword by keyword. Aliases are written on the
interface, and the first interface is the output, as before.

```julia
# v6
@node MyGaussian Stochastic [out, (μ, aliases = [mean]), (τ, aliases = [precision])]
```

```@example v7
using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
import MessagePassingRulesBase: annotate!, getannotation

struct MyGaussian end

@define_factor_node(
    node = MyGaussian,
    type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), (:τ, aliases = [:precision])],
)
```

A node with a variable number of edges of one kind, which v6 wrote by hand with `ManyOf` and a
node type of its own, declares an interface group, `:in...`; see [Groups](@ref migration-v6-to-v7-groups).

## Message update rules

`@rule` becomes `@define_message_update_rule`. The node, the target and the inputs are keywords;
the inputs are typed like the old arguments, with `m_x` becoming `m[:x]` and `q_x` becoming
`q[:x]`; the body is a lambda over the slots it needs. `Marginalisation` is gone: a rule belongs
to an algorithm, the node's default unless it says otherwise.

```julia
# v6
@rule MyGaussian(:out, Marginalisation) (m_μ::PointMass, m_τ::PointMass) = begin
    return NormalMeanPrecision(mean(m_μ), mean(m_τ))
end

@rule MyGaussian(:out, Marginalisation) (q_μ::Any, q_τ::Any) = NormalMeanPrecision(mean(q_μ), mean(q_τ))
```

```@example v7
@define_message_update_rule(
    node = MyGaussian, target = :out,
    args = (m[:μ]::PointMass, m[:τ]::PointMass),
    body = (args) -> NormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:τ])),
)

@define_message_update_rule(
    node = MyGaussian, target = :out,
    args = (q[:μ]::Any, q[:τ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:τ])),
)

call_message_update_rule(MyGaussian, :out; q = (μ = NormalMeanVariance(1.0, 1.0), τ = GammaShapeRate(2.0, 1.0)))
```

A joint marginal, `q_out_μ` in v6, is `q[:out, :μ]`, its members in interface order:

```julia
# v6
@rule MyGaussian(:τ, Marginalisation) (q_out_μ::Any,) = begin
    m, V = mean_cov(q_out_μ)
    return GammaShapeRate(3 / 2, (V[1, 1] - V[1, 2] - V[2, 1] + V[2, 2] + abs2(m[1] - m[2])) / 2)
end
```

```@example v7
@define_message_update_rule(
    node = MyGaussian, target = :τ,
    args = (q[:out, :μ]::Any,),
    body = (args) -> begin
        m, V = mean_cov(args.q[:out, :μ])
        GammaShapeRate(3 / 2, (V[1, 1] - V[1, 2] - V[2, 1] + V[2, 2] + abs2(m[1] - m[2])) / 2)
    end,
)

call_message_update_rule(MyGaussian, :τ; clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.0; 0.0 1.0]),))
```

## Marginal rules

`@marginalrule` becomes `@define_marginal_update_rule`, its target the cluster's members as a
tuple rather than a joined name (`:out_μ` becomes `(:out, :μ)`). A result that factorises into
independent blocks, which v6 returned as a NamedTuple, is a [`FactorizedCluster`](@ref), each
block labelled with the members it covers.

```julia
# v6
@marginalrule MyGaussian(:out_μ) (m_out::PointMass, m_μ::NormalMeanPrecision, q_τ::Any) = begin
    return (out = m_out, μ = prod(ClosedProd(), NormalMeanPrecision(mean(m_out), mean(q_τ)), m_μ))
end
```

```@example v7
@define_marginal_update_rule(
    node = MyGaussian, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::NormalMeanPrecision, q[:τ]::Any),
    body = (args) -> FactorizedCluster(
        (:out,) => args.m[:out],
        (:μ,) => prod(ClosedProd(), NormalMeanPrecision(mean(args.m[:out]), mean(args.q[:τ])), args.m[:μ]),
    ),
)

call_marginal_update_rule(MyGaussian, (:out, :μ); m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0)), q = (τ = PointMass(1.0),))
```

## Average energies

`@average_energy` becomes `@define_average_energy`, with the same argument syntax as the rules.

```julia
# v6
@average_energy MyGaussian (q_out::Any, q_μ::Any, q_τ::Any) = begin
    return (log(2π) - mean(log, q_τ) + mean(q_τ) * (var(q_out) + var(q_μ) + abs2(mean(q_out) - mean(q_μ)))) / 2
end
```

```@example v7
@define_average_energy(
    node = MyGaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::Any),
    body = (args) -> (log(2π) - mean(log, args.q[:τ]) + mean(args.q[:τ]) * (var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ])))) / 2,
)

call_average_energy(MyGaussian; q = (out = PointMass(1.0), μ = PointMass(0.0), τ = PointMass(1.0)))
```

## Log scales and annotations

`@logscale v` becomes `annotate!(ann, :logscale, v)` on the rule's `ann` slot. A rule reads the
log scales its inputs arrived with from `ann.m[:x]` (and `ann.q[:x]`), where v6 read them from
the raw `messages` tuple.

Log scales are experimental in v7, as they were in effect in v6: their gaps are kept, not
fixed, and a later release may change how they are carried or remove them. Port a rule's log
scale when it has one, but do not build new functionality on them.

```julia
# v6
@rule MyBernoulli(:p, Marginalisation) (m_out::PointMass,) = begin
    @logscale -log(2)
    return Beta(1 + mean(m_out), 2 - mean(m_out))
end
```

```@example v7
struct MyBernoulli end

@define_factor_node(node = MyBernoulli, type = Stochastic, interfaces = [:out, :p])

@define_message_update_rule(
    node = MyBernoulli, target = :p,
    args = (m[:out]::PointMass,),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, -log(2))
        Beta(1 + mean(args.m[:out]), 2 - mean(args.m[:out]))
    end,
)

store = MessagePassingRulesBase.AnnotationStore()
call_message_update_rule(MyBernoulli, :p; m = (out = PointMass(1.0),), ann = store), getannotation(store, :logscale)
```

## [Groups](@id migration-v6-to-v7-groups)

A variable number of edges of one kind, `ManyOf` in v6, is an interface group. A rule towards a
member names it `(:in, k)`, and its inputs select members: `m[:in...]` all of them, `m[:in][k]`
the target's own, and `m[:in][!k]` all but it. A group arrives as a tuple in member order, with
`nothing` for a member the rule does not take. The node needs no node type, `factornode` or
`activate!` of its own, and no `{N}` parameter: the number of members is the group's length.

```julia
# v6
@rule MySum{N}(:out, Marginalisation) (m_in::ManyOf{N, NormalMeanVariance},) where {N} = begin
    return NormalMeanVariance(sum(mean, m_in), sum(var, m_in))
end
```

```@example v7
struct MySum end

@define_factor_node(node = MySum, type = Deterministic, interfaces = [:out, :in...])

@define_message_update_rule(
    node = MySum, target = :out,
    args = (m[:in...]::NormalMeanVariance,),
    body = (args) -> NormalMeanVariance(sum(mean, args.m[:in]), sum(var, args.m[:in])),
)

@define_message_update_rule(
    node = MySum, target = (:in, k),
    args = (m[:out]::NormalMeanVariance, m[:in][!k]::NormalMeanVariance),
    body = (args) -> begin
        others = [m for m in args.m[:in] if m !== nothing]
        NormalMeanVariance(mean(args.m[:out]) - sum(mean, others), var(args.m[:out]) + sum(var, others))
    end,
)

call_message_update_rule(MySum, (:in, 1); m = (out = NormalMeanVariance(3.0, 1.0), in = (nothing, NormalMeanVariance(2.0, 1.0))))
```

## `meta`

`meta` served two purposes, and each has its own place now.

- **What a rule computes**, such as an approximation method, is an **algorithm**: a type
  `struct MyMethod <: AbstractAlgorithm end`, possibly with fields, that the rule names with
  `algorithm = MyMethod` and receives in its `algo` slot. The node's user chooses it per node, as
  they chose the meta. `DeltaMeta(method = Unscented())` is `DeltaApproximation(method = Unscented())`,
  and `DeltaMeta(method = Linearization(), inverse = f⁻¹)` is `DeltaApproximation(method =
  Linearization(), inverse = f⁻¹)`; `Unscented` and `Linearization` come from
  `MessagePassingRulesApproximations`.
- **How a rule computes it numerically**, such as a matrix correction or a random number
  generator, is a **context service**, declared with `ctx = (:matrix_correction,)` and read as
  `matrix_correction(ctx, default)` or `ctx.rng`. `default_meta` becomes the rule's `default`.

## Functional dependencies

v6's functional dependencies become declared dependencies. The default scheme, which gives a
rule the messages in its own cluster and the marginals of the others, needs no declaration.

| v6 | v7 |
|---|---|
| `DefaultFunctionalDependencies` | nothing: the default |
| a node with its own `functional_dependencies` | an algorithm of the node's own, with `dependencies = [...]` on `@define_factor_node` |
| `RequireMessageFunctionalDependencies`, `RequireMarginalFunctionalDependencies`, `RequireEverythingFunctionalDependencies` | for one model, a [`DefaultAlgorithmExtension`](@ref) with its own dependencies (`@define_dependencies`); for a node, its own algorithm |
| `RequireMarginalFunctionalDependencies(a = nothing)`, keeping the default and adding `q(a)` | `:a => (default, q[:a])`, with `default` alone for the other targets ([Extending the default scheme](@ref rules-algorithms-extending)) |
| RxInfer's `where { dependencies = … }` | choosing the node's algorithm |

A target's inputs are subscribed to in the order they are declared, which is the update
schedule in variational message passing.

## Calling rules, and other renames

| v6 | v7 |
|---|---|
| `@call_rule Node(:out, Marginalisation) (m_x = …,)` | `call_message_update_rule(Node, :out; m = (x = …,))`, or `@call_message_update_rule` |
| `@call_marginalrule` | `call_marginal_update_rule`, `@call_marginal_update_rule` |
| `score(AverageEnergy(), Node, Val{…}(), marginals, meta)` | `call_average_energy(Node; q = …)` |
| a rule calling another rule | both calling a plain helper function |
| `to_marginal(d)` | `public_equivalent(d)`, a method a package adds for its working types |
| `getnodefn(node)`, `getnode()` in a rule | `getnodefn(ctx.node, target)`, `ctx.node` |
| `@test_rules` | `@test_message_update_rule` ([Testing rules](@ref rules-testing)) |

## [What cannot be translated mechanically](@id migration-v6-to-v7-manual)

These need a person who knows what the rule means. Stop and ask.

- **Rules reading the raw `messages` or `marginals` tuple**, often for their annotations. Each
  use has to be matched to a named input or to `ann.m`/`ann.q`, which depends on what the rule
  meant by the index.
- **Rules building graph objects**, such as a `randomvar` for a product with a log scale. A
  product is the `ctx.product` service; anything else has no counterpart.
- **`meta` used as mutable workspace**, such as a cache filled across calls. A rule is pure unless
  it says `pure = false`; state belongs to an algorithm that declares itself impure, and whether
  that is right depends on the model.
- **Rule fallbacks** are not carried over: when no rule fits, the base package reports the closest
  candidates.

## [Verifying a port](@id migration-v6-to-v7-verify)

A ported rule is checked three ways:

1. A table of cases, [`@test_message_update_rule`](@ref), with values derived by hand or taken
   from the old tests.
2. Where the inputs allow, [`@verify_message_update_rule`](@ref), which checks a message against
   the node's definition.
3. While the old implementation is at hand, [`compare_with_reference`](@ref) on the same inputs:
   every difference is either a bug in the port or a declared correction with its reason.

## [Node packages](@id migration-v6-to-v7-node-packages)

The nodes that are not standard have a package each, loaded next to the engine: loading it is
enough for the engine to find its rules. Their v6 `meta` is the node's own algorithm:

| v6 | v7 |
|---|---|
| `GaussianCoupling` | `GaussianCouplingMessagePassingRules`, no algorithm of its own |
| `Probit`, `ProbitMeta(p)` | `ProbitMessagePassingRules`, `ProbitEP(; p = 32)` |
| `GCV`, `GCVMetadata(GaussHermiteCubature(n))` | `GCVMessagePassingRules`, `GCVApproximation(; method = GaussHermiteCubature(n))` |
| `AR`, `ConjugateAR`, `ARMeta(form, order, stype)` | `AutoregressiveMessagePassingRules`, `ARVMP(form, order, stype)` with `ARsafe()` or `ARunsafe()` |
| `SoftDot` (`softdot`) | `SoftDotMessagePassingRules`, no algorithm of its own |
| `ContinuousTransition` (`CTransition`), `CTMeta(f)` | `ContinuousTransitionMessagePassingRules`, `CTVMP(f)` |
| `BinomialPolya`, `BinomialPolyaMeta(n, rng)` | `PolyaMessagePassingRules`, `BinomialPolyaApproximation(; samples = n)`, drawing from the engine's generator |
| `MultinomialPolya`, `MultinomialPolyaMeta(points)` | `PolyaMessagePassingRules`, `MultinomialPolyaApproximation(; points)` |
| `BIFM`, `BIFMHelper`, `BIFMMeta(A, B, C)` | `BIFMMessagePassingRules`, `BIFMSmoother(A, B, C)` |
| `Flow`, `FlowMeta(model, approximation)` | `FlowMessagePassingRules`, `FlowApproximation(model; method = approximation)` |

- **Probit** declared `RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0, 100))`. Its
  algorithm now declares that the rule towards `in` reads the message on its own edge, and the
  node declares that message's start, set only where the model sets none. A model that chose
  another initial message keeps it. v6's rules without expectation propagation run under
  `DefaultAlgorithm()`.
- **GCV**'s `ExponentialLinearQuadratic` is exported by its package, which also holds the rules
  that let `NormalMeanVariance` and `NormalMeanPrecision` take one on `out`.
- **AR** and **ConjugateAR** declare no algorithm, as v6 had no default `ARMeta`. A model gives
  each node `ARVMP(...)`, and without one no rule is found. ConjugateAR's marginal over `w` alone
  is the engine's product of its messages, as for any single interface.
- **SoftDot** does not need the AR package; loading its own package is enough.
- **ContinuousTransition** declares no algorithm, as v6 had no default `CTMeta`: a model gives
  each node `CTVMP(f)`. Its rule towards `a` still reads `q(a)`, as
  `RequireMarginalFunctionalDependencies(a = nothing)` made it, now through its declaration
  ([Extending the default scheme](@ref rules-algorithms-extending)). As in v6, the node sets no
  initial `q(a)`; a model initialises it.
- **BinomialPolya** and **MultinomialPolya** read the message on their weights' own edge, which v6
  required each model to ask for with `where { dependencies = RequireMessageFunctionalDependencies(β
  = …) }`. The nodes now declare it; drop the `dependencies` and initialise the message instead,
  `μ(β) = …` in RxInfer's `@initialization`. Their package is GPL-3 licensed, through
  PolyaGammaHybridSamplers.
- **BIFM** keeps nothing between calls. v6's `BIFMMeta` was a cache its rules shared, so they had
  to run in a set order and each node needed a meta of its own. `BIFMSmoother` is an ordinary
  value that nodes may share, and the order the posteriors are subscribed in no longer matters.
  `BIFMMeta(A, B, C, μu, Σu)` has no counterpart: the input's statistics come from its message.
- **Flow**'s models, layers and `PermutationMatrix` live in its package. `ReactiveMP.forward(model, x)`
  and its siblings are `FlowMessagePassingRules.forward`, public and unexported. Building a
  model that draws takes a generator first, `compile(rng, model)`, `PermutationMatrix(rng, dim)`,
  and without one draws from the task's as v6 did. `Unscented()` needs no dimension.

## Behaviour that changed

The rules of the standard nodes follow naive variational message passing where v6 did not, and
fix errors v6 had. A result that differs from v6's for these nodes is expected:

- **Variational rules** take a variance's contribution as `1/E[1/v]` and a covariance's as
  `E[Σ⁻¹]⁻¹` (NormalMeanVariance, MvNormalMeanCovariance), where v6 took `E[v]` and `E[Σ]`.
- **Average energies** of Gamma and GammaInverse compute `E[x]·E[1/θ]` and `θ·E[1/x]`, where v6
  divided expectations.
- **Wishart**'s variational rule towards `out` uses `E[S⁻¹]`; **MvNormalGamma**'s includes the
  covariance of `μ` in the rate; **MvNormalWeightedMeanPrecision**'s marginal labels its blocks
  correctly.
- **Arithmetic:** `+` and `-` fix sign errors for weighted-mean normals and for `-`'s marginal;
  `*`'s sampled messages towards a factor drop a spurious weight, its log scale towards `in` is
  `-d·log|a|`, and a product that would need `in * A` for a matrix operand has no rule. v6's
  `*` rules towards `in` with their arguments reversed, reachable only through `@call_rule`, are
  gone.
- **Mixture** has no average energy: the free energy of a model with one is an error, not zero.
- **Probit's average energy** is finite for a wide `q(in)`, where v6's underflowed at far cubature
  points and returned Inf or NaN.
- **ContinuousTransition's average energies** are the closed form; v6's were wrong in three
  terms, so a model's free energy changes with them. Its rules towards `a` and `W` keep the offset
  of an affine or nonlinear `f`, such as a rotation, which v6 dropped; for `reshape` nothing
  changes.
- **The Pólya nodes' average energies** are corrected: BinomialPolya's is the expectation of
  `softplus(xᵀβ)`, where v6 took it at the mean, and MultinomialPolya's is right for a Multinomial
  `q(x)` with more than one trial. A binomial regression's free energy is higher than v6's.
- **The free energy of a model with BIFM** raises an error naming the node, where v6's failed with
  an infinite node bound.
- **`ARunsafe`'s joint `q(y, x)`** is correct: v6's disagreed with `ARsafe` even for an AR(1), and
  threw for a multivariate AR. `ARsafe` is unchanged.
- **The Delta node takes three methods**: `Unscented()`, `Linearization()` and, once
  `using ExponentialFamilyProjection` loads its rules, `CVIProjection()`. v6's other methods are
  gone, with no replacement: `CVI` and `ProdCVI`, `LaplaceApproximation`,
  `ImportanceSamplingApproximation`, `GaussLaguerreQuadrature` and the spherical-radial cubature.
  `DeltaApproximation` refuses any other method with an error that lists these three and, for
  `CVIProjection` without its package, says which package to load. `CVIProjection` has no `rng`
  field: it samples from the generator the engine gives the rule, and its joint rule, which
  keeps its result as the next proposal, is impure. With several inputs, that rule projects them
  in turn, each against the others' latest projections, where v6 used the previous proposal for
  all: its results differ from v6's, and on a posterior with several modes it settles on one
  where v6's alternated.

## Removed

These v6 names have no counterpart: `Marginalisation`, `MomentMatching`, the functional
dependency types, the per-node node types (`NormalMixtureNode`, `GammaMixtureNode`,
`MixtureNode`; the checks their constructors made, at least two components, as many of each
kind, a mean-field factorisation, are the `matched_groups`, `min_group_length` and
`factorisation` of [`@define_factor_node`](@ref)), `NodeFunctionRuleFallback`, and the approximation methods
with no remaining consumer (`CVI`, `ProdCVI`, `Adam`, `ForwardDiffGrad`, `LaplaceApproximation`,
`ImportanceSamplingApproximation`, `GaussLaguerreQuadrature`, `srcubature`).
