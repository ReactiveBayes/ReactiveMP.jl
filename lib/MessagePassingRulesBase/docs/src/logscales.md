```@meta
CurrentModule = MessagePassingRulesBase
```

# Log scales

A [message](@ref glossary-message)'s [log scale](@ref glossary-log-scale) is a scalar carried
beside it. A rule returns a normalised distribution, but what it computes may be an unnormalised
function. The log scale is the constant that relates the two:

```math
\text{message}(x) = \exp(\text{logscale}) \cdot \hat{p}(x),
```

where ``\hat{p}`` is the distribution the rule returns.

[Belief propagation](@ref glossary-belief-propagation) is the common case. A message
``\mu(x) = \int f(x, y, \dots) \prod_i \mu_i(y_i) \,\mathrm{d}y`` is in general not normalised.
In an acyclic graph, the log scales of such messages add up to the log model evidence, which an
engine can then read at any edge. A naive [variational](@ref glossary-vmp) message,
``\exp \mathbb{E}_q[\log f]``, has no constant with that meaning, so its log scale is undefined.

This page covers the rule's side: what a rule declares, and how it reads the log scales of its
inputs. The engine's documentation covers how an engine propagates log scales through products,
and what the evidence means.

## What a rule declares

The normalised distribution a rule returns does not reveal its log scale. `Beta(2, 1)` looks the
same whether or not a constant was divided out to get it. A message rule therefore states its
log scale with the `logscale` keyword of [`@define_message_update_rule`](@ref), in one of four
forms:

| form | when to use it |
|---|---|
| a constant, `logscale = 0` | the rule's constant does not depend on its inputs |
| a function of the inputs, `logscale = (args) -> …` | the constant depends on the inputs |
| `logscale = from_body` | the constant shares its work with the result |
| the keyword omitted | the log scale is undefined, as for every variational rule |

A marginal rule and an average energy have no log scale, and they do not accept the keyword.

### A constant

Zero is the common constant. The convolution of normals integrates to one. A stochastic node's
message towards `out` from point-mass inputs is its own normalised density.

Zero is a fact about the rule, not a safe default. A Bernoulli node with its output observed at
1 sends ``p \mapsto p`` towards `p`. That function integrates to ``\tfrac{1}{2}`` over
``[0, 1]``, so the rule returns `Beta(2, 1)` and declares `logscale = loghalf`.

An `Irrational` constant, or an integer, takes the precision of whatever it is added to. A
`Float32` model therefore stays `Float32`.

### A function of the inputs

The function takes some of the slots `(algo, ctx, args)`, named in that order. The node below
scales its input, `out = a · in`. Its message towards `in` is the normal message on `out`
evaluated at ``a x``, which integrates to ``1 / |a|``:

```math
\mathcal{N}(a x \mid m, v) = \frac{1}{|a|}\, \mathcal{N}\big(x \mid m / a,\, v / a^2\big).
```

```@example logscales-function
using MessagePassingRulesBase, BayesBase, ExponentialFamily

struct Scale end   # out = a · in

@define_factor_node(node = Scale, type = Deterministic, interfaces = [:out, :in, :a])

@define_message_update_rule(
    node = Scale, target = :in, args = (m[:out]::NormalMeanVariance, m[:a]::PointMass),
    logscale = (args) -> -log(abs(mean(args.m[:a]))),
    body = (args) -> begin
        a = mean(args.m[:a])
        NormalMeanVariance(mean(args.m[:out]) / a, var(args.m[:out]) / a^2)
    end,
)

@call_message_update_rule(
    node = Scale, target = :in,
    m = (out = NormalMeanVariance(4.0, 1.0), a = PointMass(2.0)),
)
```

The result's log scale is ``-\log 2``.

### From the body, or none

With `logscale = from_body`, the body returns [`with_logscale`](@ref)`(result, logscale)`, and
whoever runs the rule unwraps it. A rule that omits the keyword gives its message an
[`UndefinedLogScale`](@ref) that names the rule.

```jldoctest logscales
julia> using MessagePassingRulesBase

julia> struct Halve end

julia> @define_factor_node(node = Halve, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Halve, target = :out, args = (m[:in]::Real,),
           logscale = from_body,
           body = (args) -> with_logscale(args.m[:in] / 2, log(2)),
       )

julia> @define_message_update_rule(node = Halve, target = :in, args = (m[:out]::Real,), body = (args) -> 2 * args.m[:out])

julia> result = @call_message_update_rule(node = Halve, target = :out, m = (in = 3.0,));

julia> getresult(result), getlogscale(result) ≈ log(2)
(1.5, true)

julia> getlogscale(@call_message_update_rule(node = Halve, target = :in, m = (out = 1.0,)))
UndefinedLogScale: the message rule for Halve towards :in under DefaultAlgorithm declares no `logscale`
```

## Undefined log scales

An undefined log scale records why it is undefined, and it propagates. Adding it to a number, or
to another undefined log scale, gives an undefined log scale with the first reason. Nothing
errors because a log scale is undefined. The exception is a rule or a user that asks for its
value with [`require_logscale`](@ref).

```@docs
MessagePassingRulesBase.UndefinedLogScale
MessagePassingRulesBase.UndefinedLogScaleError
MessagePassingRulesBase.require_logscale
MessagePassingRulesBase.isdefined_logscale
MessagePassingRulesBase.getlogscale
```

## Declaring a log scale from the body

A body that computes the result and the constant together returns both with
[`with_logscale`](@ref). The rule declares `logscale = from_body`, as `Halve` does above.

```@docs
MessagePassingRulesBase.with_logscale
MessagePassingRulesBase.from_body
```

## Reading the inputs' log scales

A rule that needs the log scales its inputs arrived with declares `reads_logscale = true`. It
reads them as `args.logscale.m[:x]`, keyed like `args.m`. It calls [`require_logscale`](@ref) on
those whose value it needs.

The rule's caller must provide the log scales, or the call is an error. An engine provides them
when it tracks log scales, as ReactiveMP does with its activation option `logscales = true`. A
call by hand takes them as `logscale = (x = …,)`.

The `Mixture` rules are such rules: the message towards the switch is a softmax over the
evidence of each component.

```jldoctest logscales
julia> struct Evidence end

julia> @define_factor_node(node = Evidence, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Evidence, target = :out, args = (m[:in]::Real,),
           reads_logscale = true,
           body = (args) -> require_logscale(args.logscale.m[:in]),
       )

julia> getresult(@call_message_update_rule(node = Evidence, target = :out, m = (in = 1.0,), logscale = (in = -2.5,)))
-2.5
```

```@docs
MessagePassingRulesBase.RuleLogScales
```
