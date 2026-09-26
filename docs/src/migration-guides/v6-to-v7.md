# [Migrating from v6 to v7](@id migration-v6-to-v7)

ReactiveMP v7 moves nodes and rules out of the engine into packages of their own. The engine
keeps variables, factor nodes, messages, marginals and the free energy; `MessagePassingRulesBase`
defines how nodes and rules are declared and found; `StandardMessagePassingRules` holds the
standard nodes' rules; `MessagePassingRulesApproximations` and `DeltaMessagePassingRules` hold the
approximation methods and the Delta node; and every other node has a package of its own
([Node packages](@ref migration-v6-to-v7-node-packages)). Most of a port is mechanical, and this guide lists the
mechanical translations as before/after pairs. The v7 side of each pair runs when these docs are
built.

## For an agent porting code

Read this section before changing anything.

- **Read first:** the macros of [MessagePassingRulesBase](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/),
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node),
  [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule) and
  [`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies), then the pairs below.
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

A node is declared with [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node), keyword by keyword. Aliases are written on the
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

`@rule` becomes [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule). The node, the target and the inputs are keywords;
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

getresult(call_message_update_rule(MyGaussian, :out; q = (μ = NormalMeanVariance(1.0, 1.0), τ = GammaShapeRate(2.0, 1.0))))
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

getresult(call_message_update_rule(MyGaussian, :τ; clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.0; 0.0 1.0]),)))
```

## Marginal rules

`@marginalrule` becomes [`@define_marginal_update_rule`](@extref MessagePassingRulesBase.@define_marginal_update_rule), its target the cluster's members as a
tuple rather than a joined name (`:out_μ` becomes `(:out, :μ)`). A result that factorises into
independent blocks, which v6 returned as a NamedTuple, is a [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster), each
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

getresult(call_marginal_update_rule(MyGaussian, (:out, :μ); m = (out = PointMass(1.0), μ = NormalMeanPrecision(0.0, 1.0)), q = (τ = PointMass(1.0),)))
```

## Average energies

`@average_energy` becomes [`@define_average_energy`](@extref MessagePassingRulesBase.@define_average_energy), with the same argument syntax as the rules.

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

getresult(call_average_energy(MyGaussian; q = (out = PointMass(1.0), μ = PointMass(0.0), τ = PointMass(1.0))))
```

## Log scales

A message's log scale is no longer an annotation: it is part of the message, the scalar with
`message = exp(logscale) · distribution` (see [Log scales](@ref lib-logscale)). A rule declares
it with the `logscale` keyword instead of writing it in its body: `@logscale v` becomes
`logscale = v` for a constant, `logscale = (args) -> …` for one computed from the inputs, or
`logscale = from_body` with the body returning [`with_logscale`](@extref MessagePassingRulesBase.with_logscale)`(result, v)`. A rule that reads the
log scales its inputs arrived with, which v6 read from the raw `messages` tuple, declares
`reads_logscale = true` and reads `args.logscale.m[:x]`.

The engine tracks them with the activation option `logscales = true`, which replaces
`LogScaleAnnotations`; RxInfer's `infer(...; logscales = true)` replaces
`annotations = LogScaleAnnotations()`, and `getlogscale(getannotations(q))` becomes
`getlogscale(q)`. A rule that declares none no longer makes inference fail: its message's log
scale is an [`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale) saying why, which propagates through products, and only a rule or
a user that needs the number errors. v6's fallback, which set zero whenever every input was a
point mass, is gone, since it was not right for every rule; the rules it covered declare their log
scale.

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
    logscale = -log(2),
    body = (args) -> Beta(1 + mean(args.m[:out]), 2 - mean(args.m[:out])),
)

result = call_message_update_rule(MyBernoulli, :p; m = (out = PointMass(1.0),))
getresult(result), getlogscale(result)
```

## Calling a rule

`@call_rule` returned the message, or a tuple with its add-ons under an option. Every call now
returns a [`RuleResult`](@extref MessagePassingRulesBase.RuleResult), the same shape whatever the rule: [`getresult`](@extref MessagePassingRulesBase.getresult)`(result)` is the message,
`getlogscale(result)` its log scale, and [`getrule`](@extref MessagePassingRulesBase.getrule), `MessagePassingRulesBase.getalgorithm` and
the other getters say what produced it.

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

getresult(call_message_update_rule(MySum, (:in, 1); m = (out = NormalMeanVariance(3.0, 1.0), in = (nothing, NormalMeanVariance(2.0, 1.0)))))
```

A group may be empty where the node declares `min_group_length = 0`. A joint may hold some of a
group's members with other interfaces, keyed with the members, `(:out, (:T, 1))`, and read as
`q[:out, (:T, 1)]`. v6 handed such joints to hand-written `rule` methods under mangled names,
`q_out_T1`, which the rule parsed. A rule that takes whatever inputs the factorisation delivers,
as those did, is written with `default` in its arguments, and walks its inputs by key with
[`MessagePassingRulesBase.rule_inputs`](@extref): an interface by its name, a member as `(:T, k)`,
a joint by its key.

```julia
# v6
function ReactiveMP.rule(::Type{<:MyTensor}, ::Val{:out}, ::Marginalisation, mnames, messages, qnames, marginals, meta, annotations, node)
    # split names such as `:in_T1` on `_` to find which edges each input covers
end
```

```@example v7
struct MyTensor end

@define_factor_node(node = MyTensor, type = Stochastic, interfaces = [:out, :in, :T...], min_group_length = 0)

@define_message_update_rule(
    node = MyTensor, target = :out, args = (default,),
    body = (args) -> sum(last, MessagePassingRulesBase.rule_inputs(MyTensor, args.q)),
)

# Under q(out) q(in, T1) q(T2): the joint by its key, and the second member of `T`.
getresult(call_message_update_rule(MyTensor, :out; clusters = ((:in, (:T, 1)) => 0.5,), q = (T = (nothing, 1.0),)))
```

## `meta`

`meta` served two purposes, and each has its own place now.

- **What a rule computes**, such as an approximation method, is an **algorithm**: a type
  `struct MyMethod <: AbstractAlgorithm end`, possibly with fields, that the rule names with
  `algorithm = MyMethod` and receives in its `algo` slot. The node's user chooses it per node, as
  they chose the meta. `DeltaMeta(method = Unscented())` is [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation)`(method = Unscented())`,
  and `DeltaMeta(method = Linearization(), inverse = f⁻¹)` is `DeltaApproximation(method =
  Linearization(), inverse = f⁻¹)`; [`Unscented`](@extref MessagePassingRulesApproximations.Unscented) and [`Linearization`](@extref MessagePassingRulesApproximations.Linearization) come from
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
| a node with its own `functional_dependencies` | an algorithm of the node's own, with `dependencies = [...]` on [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) |
| `RequireMessageFunctionalDependencies`, `RequireMarginalFunctionalDependencies`, `RequireEverythingFunctionalDependencies` | for one model, a [`DefaultAlgorithmExtension`](@extref MessagePassingRulesBase.DefaultAlgorithmExtension) with its own dependencies ([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)); for a node, its own algorithm |
| `RequireMarginalFunctionalDependencies(a = nothing)`, keeping the default and adding `q(a)` | `:a => (default, q[:a])`, with `default` alone for the other targets ([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)) |
| RxInfer's `where { dependencies = … }` | choosing the node's algorithm |

A target's inputs are subscribed to in the order they are declared, which is the update
schedule in variational message passing.

## Calling rules, and other renames

| v6 | v7 |
|---|---|
| `@call_rule Node(:out, Marginalisation) (m_x = …,)` | [`getresult`](@extref MessagePassingRulesBase.getresult)`(call_message_update_rule(Node, :out; m = (x = …,)))`, or [`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) |
| `@call_marginalrule` | [`call_marginal_update_rule`](@extref MessagePassingRulesBase.call_marginal_update_rule), [`@call_marginal_update_rule`](@extref MessagePassingRulesBase.@call_marginal_update_rule) |
| `score(AverageEnergy(), Node, Val{…}(), marginals, meta)` | [`call_average_energy`](@extref MessagePassingRulesBase.call_average_energy)`(Node; q = …)` |
| a rule calling another rule | both calling a plain helper function |
| `to_marginal(d)` | [`public_equivalent`](@extref MessagePassingRulesBase.public_equivalent)`(d)`, a method a package adds for its working types |
| `getnodefn(node)`, `getnode()` in a rule | [`getnodefn`](@extref MessagePassingRulesBase.getnodefn)`(ctx.node, target)`, `ctx.node` |
| `nodefunction(node, meta, Val(:out))`; a known inverse as `nodefunction(node, meta, (Val(:in), k))` | `getnodefn(ctx.node, Target(:out))`; an inverse is the algorithm's, read from `algo` |
| `@test_rules` | [`@test_message_update_rule`](@extref MessagePassingRulesTestUtils.@test_message_update_rule) |
| [`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback)`()` as the engine's `rulefallback` | the same, from `MessagePassingRulesBase`, as the activation option `rulefallback` ([Rule fallbacks](@ref lib-activation-options-rulefallback)); its message is a [`NodeFunctionLogPdf`](@extref MessagePassingRulesBase.NodeFunctionLogPdf) |

## [What cannot be translated mechanically](@id migration-v6-to-v7-manual)

These need a person who knows what the rule means. Stop and ask.

- **Rules reading the raw `messages` or `marginals` tuple**, often for their annotations or log
  scales. Each use has to be matched to a named input, to `ann.m`/`ann.q` or to
  `args.logscale.m`, which depends on what the rule meant by the index.
- **Rules building graph objects**, such as a `randomvar` for a product with a log scale. The log
  scale of a product of two distributions is `BayesBase.compute_logscale` of it, as `Mixture`'s
  rule towards its switch computes it; anything else has no counterpart.
- **`meta` used as mutable workspace**, such as a cache filled across calls. A rule is pure unless
  it says `pure = false`; state belongs to an algorithm that declares itself impure, and whether
  that is right depends on the model.

## [Verifying a port](@id migration-v6-to-v7-verify)

A ported rule is checked three ways:

1. A table of cases, [`@test_message_update_rule`](@extref MessagePassingRulesTestUtils.@test_message_update_rule), with values derived by hand or taken
   from the old tests.
2. Where the inputs allow, [`@verify_message_update_rule`](@extref MessagePassingRulesTestUtils.@verify_message_update_rule), which checks a message against
   the node's definition.
3. While the old implementation is at hand, [`compare_with_reference`](@extref MessagePassingRulesTestUtils.compare_with_reference) on the same inputs:
   every difference is either a bug in the port or a declared correction with its reason.

## [Node packages](@id migration-v6-to-v7-node-packages)

The nodes that are not standard have a package each, loaded next to the engine: loading it is
enough for the engine to find its rules. Their v6 `meta` is the node's own algorithm:

| v6 | v7 |
|---|---|
| `GaussianCoupling` | [GaussianCouplingMessagePassingRules](https://reactivebayes.github.io/GaussianCouplingMessagePassingRules.jl/dev/), no algorithm of its own |
| `Probit`, `ProbitMeta(p)` | [ProbitMessagePassingRules](https://reactivebayes.github.io/ProbitMessagePassingRules.jl/dev/), [`ProbitEP`](@extref ProbitMessagePassingRules.ProbitEP)`(; p = 32)` |
| `GCV`, `GCVMetadata(GaussHermiteCubature(n))` | [GCVMessagePassingRules](https://reactivebayes.github.io/GCVMessagePassingRules.jl/dev/), [`GCVApproximation`](@extref GCVMessagePassingRules.GCVApproximation)`(; method = GaussHermiteCubature(n))` |
| `AR`, `ConjugateAR`, `ARMeta(form, order, stype)` | [AutoregressiveMessagePassingRules](https://reactivebayes.github.io/AutoregressiveMessagePassingRules.jl/dev/), [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP)`(form, order, stype)` with [`ARsafe`](@extref AutoregressiveMessagePassingRules.ARsafe)`()` or [`ARunsafe`](@extref AutoregressiveMessagePassingRules.ARunsafe)`()` |
| `SoftDot` (`softdot`) | [SoftDotMessagePassingRules](https://reactivebayes.github.io/SoftDotMessagePassingRules.jl/dev/), no algorithm of its own |
| `ContinuousTransition` (`CTransition`), `CTMeta(f)` or `ContinuousTransitionMeta(f)` | [ContinuousTransitionMessagePassingRules](https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/dev/), [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP)`(f)` |
| `BinomialPolya`, `BinomialPolyaMeta(n, rng)` | [PolyaMessagePassingRules](https://reactivebayes.github.io/PolyaMessagePassingRules.jl/dev/), [`BinomialPolyaApproximation`](@extref PolyaMessagePassingRules.BinomialPolyaApproximation)`(; samples = n)`, drawing from the engine's generator |
| `MultinomialPolya`, `MultinomialPolyaMeta(points)` | [PolyaMessagePassingRules](https://reactivebayes.github.io/PolyaMessagePassingRules.jl/dev/), [`MultinomialPolyaApproximation`](@extref PolyaMessagePassingRules.MultinomialPolyaApproximation)`(; points)` |
| `BIFM`, `BIFMHelper`, `BIFMMeta(A, B, C)` | [BIFMMessagePassingRules](https://reactivebayes.github.io/BIFMMessagePassingRules.jl/dev/), [`BIFMSmoother`](@extref BIFMMessagePassingRules.BIFMSmoother)`(A, B, C)` |
| `Flow`, `FlowMeta(model, approximation)` | [FlowMessagePassingRules](https://reactivebayes.github.io/FlowMessagePassingRules.jl/dev/), [`FlowApproximation`](@extref FlowMessagePassingRules.FlowApproximation)`(model; method = approximation)` |
| `DiscreteTransition` | [DiscreteTransitionMessagePassingRules](https://reactivebayes.github.io/DiscreteTransitionMessagePassingRules.jl/dev/), no algorithm of its own |

- **Probit** declared `RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0, 100))`. Its
  algorithm now declares that the rule towards `in` reads the message on its own edge, and the
  node declares that message's start, set only where the model sets none. A model that chose
  another initial message keeps it. v6's rules without expectation propagation run under
  [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()`.
- **GCV**'s [`ExponentialLinearQuadratic`](@extref GCVMessagePassingRules.ExponentialLinearQuadratic) is exported by its package, which also holds the rules
  that let `NormalMeanVariance` and `NormalMeanPrecision` take one on `out`.
- **AR** and **ConjugateAR** declare no algorithm, as v6 had no default `ARMeta`. A model gives
  each node [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP)`(...)`, and without one no rule is found. ConjugateAR's marginal over `w` alone
  is the engine's product of its messages, as for any single interface.
- **SoftDot** does not need the AR package; loading its own package is enough.
- **ContinuousTransition** declares no algorithm, as v6 had no default `CTMeta`: a model gives
  each node [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP)`(f)`. Its rule towards `a` still reads `q(a)`, as
  `RequireMarginalFunctionalDependencies(a = nothing)` made it, now through its declaration
  ([`@define_dependencies`](@extref MessagePassingRulesBase.@define_dependencies)). As in v6, the node sets no
  initial `q(a)`; a model initialises it.
- **BinomialPolya** and **MultinomialPolya** read the message on their weights' own edge, which v6
  required each model to ask for with `where { dependencies = RequireMessageFunctionalDependencies(β
  = …) }`. The nodes now declare it; drop the `dependencies` and initialise the message instead,
  `μ(β) = …` in RxInfer's `@initialization`. Their package is GPL-3 licensed, through
  PolyaGammaHybridSamplers.
- **BIFM** keeps nothing between calls. v6's `BIFMMeta` was a cache its rules shared, so they had
  to run in a set order and each node needed a meta of its own. [`BIFMSmoother`](@extref BIFMMessagePassingRules.BIFMSmoother) is an ordinary
  value that nodes may share, and the order the posteriors are subscribed in no longer matters.
  `BIFMMeta(A, B, C, μu, Σu)` has no counterpart: the input's statistics come from its message.
- **Flow**'s models, layers and [`PermutationMatrix`](@extref FlowMessagePassingRules.PermutationMatrix) live in its package. `ReactiveMP.forward(model, x)`
  and its siblings are `FlowMessagePassingRules.forward`, public and unexported. Building a
  model that draws takes a generator first, `compile(rng, model)`, `PermutationMatrix(rng, dim)`,
  and without one draws from the task's as v6 did. [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()` needs no dimension.
- **DiscreteTransition**'s interfaces are `out`, `in`, `a` and the group `T`, which may be empty.
  v6 took `DiscreteTransition(out, in, a, t1, t2)` positionally and aliased the extra arguments
  `T1`, `T2`; they are the members `(:T, 1)`, `(:T, 2)`, and a joint over `out` and `t1` alone is
  keyed `(:out, (:T, 1))`. Every factorisation v6 handled is handled, with one rule per target,
  none per factorisation. A marginal rule's observed members come as [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster)
  blocks, `(:out,) => …, (:in, (:T, 2)) => …`, where v6 returned `(out = …, in_T2 = …)`; one inside
  a joint over the whole group `T` stays in the joint as a one-hot axis. v6 ignored the node's
  `meta`, and it has no algorithm.
- **A distribution value as a prior**, `x ~ d` for `d = Beta(4.0, 8.0)` or `Truncated(…)`, was
  v6's `StandaloneDistributionNode`, an engine node type. It is Standard's
  [`StandaloneDistribution`](@extref StandardMessagePassingRules.StandaloneDistribution), an ordinary node `out ~ d` with `d` a constant: the message towards
  `out` is `d`, and the free-energy term `KL(q ‖ d)`, as before.

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
- **NormalMixture** and **GammaMixture** take no type parameter: v6's `NormalMixture{N}` and
  `GammaMixture{N}` are `NormalMixture` and `GammaMixture`, and the number of components is the
  length of their groups.
- **`*`** runs under [`MultiplicationSampling`](@extref StandardMessagePassingRules.MultiplicationSampling)`(; samples = 3000)`, its default algorithm: its three
  rules that sample draw `samples` values from the engine's generator, where v6 always drew 3000.
- **MatrixNormal's average energy** takes a point mass or a MatrixNormal for `q(out)` and `q(M)`,
  whose second moments it uses; v6 took any type and dropped the second moments of anything but
  a MatrixNormal. **MatrixNormalWishart's** is exact for known parameters, where v6 took
  `f(E[x])` for `E[f(x)]`.
- **MvNormalWeightedMeanPrecision's average energy** takes a Wishart `q(Λ)`, not only a point
  mass.
- **Probit's average energy** is finite for a wide `q(in)`, where v6's underflowed at far cubature
  points and returned Inf or NaN.
- **ContinuousTransition's average energies** are the closed form; v6's were wrong in three
  terms, so a model's free energy changes with them. Its rules towards `a` and `W` keep the offset
  of an affine or nonlinear `f`, such as a rotation, which v6 dropped; for `reshape` nothing
  changes.
- **The Pólya nodes' average energies** are corrected: BinomialPolya's is the expectation of
  `softplus(xᵀβ)`, where v6 took it at the mean, and MultinomialPolya's is right for a Multinomial
  `q(x)` with more than one trial. A binomial regression's free energy is higher than v6's.
  BinomialPolya's sampling takes a univariate `β` too, each draw one sample, where v6 took all the
  draws as one.
- **DiscreteTransition** normalises its five-interface belief-propagation message towards `T3`
  with a DirichletCollection `q(a)` over the whole tensor, as every other rule of the node, where
  v6 normalised over `out` only. It also takes what v6 failed on: a Bernoulli `in` or `out`, the
  energy of a joint over three or more axes with a DirichletCollection `q(a)`, and that of a
  non-square `q(out, in)` with a point-mass `q(a)`. A point-mass `A` is used as given; v6's
  belief-propagation rules clamped it to at most one, which changed nothing for a probability
  tensor. v6's fifty or so rules specialised to a number of interfaces are gone; they computed
  the generic contraction, and four of its five-interface rules towards `T2`, which read their own
  edge's message, never matched.
- **The free energy of a model with BIFM** raises an error naming the node, where v6's failed with
  an infinite node bound.
- **[`ARunsafe`](@extref AutoregressiveMessagePassingRules.ARunsafe)'s joint `q(y, x)`** is correct: v6's disagreed with [`ARsafe`](@extref AutoregressiveMessagePassingRules.ARsafe) even for an AR(1), and
  threw for a multivariate AR. `ARsafe` is unchanged.
- **The Delta node takes three methods**: [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()`, [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()` and, once
  `using ExponentialFamilyProjection` loads its rules, [`CVIProjection`](@extref DeltaMessagePassingRules.CVIProjection)`()`. v6's other methods are
  gone, with no replacement: `CVI` and `ProdCVI`, `LaplaceApproximation`,
  `ImportanceSamplingApproximation`, `GaussLaguerreQuadrature` and the spherical-radial cubature.
  [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation) refuses any other method with an error that lists these three and, for
  `CVIProjection` without its package, says which package to load. `CVIProjection` has no `rng`
  field: it samples from the generator the engine gives the rule, and its joint rule, which
  keeps its result as the next proposal, is impure. With several inputs, that rule projects them
  in turn, each against the others' latest projections, where v6 used the previous proposal for
  all: its results differ from v6's, and on a posterior with several modes it settles on one
  where v6's alternated.

## Removed

These v6 names have no counterpart: `Marginalisation`, `MomentMatching`, the functional
dependency types, the per-node node types (`NormalMixtureNode` and its alias `GaussianMixtureNode`,
`GammaMixtureNode`, `MixtureNode`; the checks their constructors made, at least two components, as many of each
kind, a mean-field factorisation, are the `matched_groups`, `min_group_length` and
`factorisation` of [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node)), and the approximation methods
with no remaining consumer (`CVI`, `ProdCVI`, `Adam` and its `update!`, `ForwardDiffGrad`,
`LaplaceApproximation` and `laplace`, `ImportanceSamplingApproximation`, `GaussLaguerreQuadrature`,
`srcubature`), with the Optimisers extension that served `ProdCVI`. MvNormalMeanPrecision's two
marginal rules for BIFM's `TerminalProdArgument` messages are gone too: only the free energy of a
BIFM model reached them, and it is not supported.

The engine no longer defines these internal helpers, none of which it used: `skip_clamped` and
`skip_clamped_and_initial` (`skip_initial` stays), `KLDivergence` and its `score` method
(`BayesBase.kldivergence` computes the divergence directly), `dropproxytype`, `other_clusters`,
`getinboundinterfaces`, `interfaceindices`, `ReactiveMP.hasfield` (which shadowed
`Base.hasfield`), `split_underscored_symbol`, `fields`, `swapped`, and the macro helpers other than
`@proxy_methods`. The v5 stubs `AddonLogScale` and `AddonMemory`, which only raised an error
pointing to their replacements, are gone too: use the `logscales = true` option and
`InputArgumentsAnnotations`.
