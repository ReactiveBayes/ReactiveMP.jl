```@meta
CurrentModule = MessagePassingRulesBase
```

# Log scales

A message's **log scale** is a scalar carried beside it. A rule returns a normalised
distribution, but what it computes may be an unnormalised function; the log scale is the
constant that relates the two:

```math
\text{message}(x) = \exp(\text{logscale}) \cdot \hat{p}(x),
```

where ``\hat{p}`` is the distribution the rule returns. Belief propagation is the common case: a
message ``\mu(x) = \int f(x, y, \dots) \prod_i \mu_i(y_i) \,\mathrm{d}y`` is in general not
normalised, and in an acyclic graph the log scales of such messages add up to the log model
evidence, which an engine can then read at any edge. A naive variational message,
``\exp \mathbb{E}_q[\log f]``, has no constant with that meaning, so its log scale is undefined.

This page is the rule's side: what a rule declares, and how it reads the log scales of its
inputs. How an engine propagates log scales through products, and what the evidence means, is
the engine's documentation.

## What a rule declares

The normalised distribution a rule returns does not reveal its log scale: `Beta(2, 1)` looks the
same whether or not a constant was divided out to get it. So a message rule states it, with the
`logscale` keyword of [`@define_message_update_rule`](@ref), in one of four forms.

- **A constant**, when the rule's constant does not depend on its inputs: `logscale = 0`. Zero is
  the common case: the convolution of normals integrates to one, and a stochastic node's message
  towards `out` from point-mass inputs is its own normalised density. It is a fact about the
  rule, not a safe default. A Bernoulli node with its output observed at 1 sends ``p \mapsto p``
  towards `p`, which integrates to ``\tfrac{1}{2}`` over ``[0, 1]``: the rule returns
  `Beta(2, 1)` and declares `logscale = loghalf`. An `Irrational` constant, or an integer, takes
  the precision of whatever it is added to, so a `Float32` model stays `Float32`.

- **A function of the inputs**, over the slots `(algo, ctx, args)`, named in that order:

  ```julia
  logscale = (args) -> -length(mean(args.m[:out])) * log(abs(mean(args.m[:A]))),
  ```

- **From the body**, `logscale = from_body`, when the constant shares its work with the result:
  the body returns [`with_logscale`](@ref)`(result, logscale)`, and whoever runs the rule unwraps
  it.

- **Nothing**, when the rule omits the keyword: its message's log scale is an
  [`UndefinedLogScale`](@ref) naming the rule. Every variational rule is such a rule.

A marginal rule and an average energy have no log scale, and do not accept the keyword.

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

An undefined log scale records why it is undefined, and propagates: adding it to a number, or to
another undefined one, gives an undefined log scale with the first reason. Nothing errors because
a log scale is undefined, except a rule or a user asking for its value with
[`require_logscale`](@ref).

```@docs
MessagePassingRulesBase.UndefinedLogScale
MessagePassingRulesBase.UndefinedLogScaleError
MessagePassingRulesBase.require_logscale
MessagePassingRulesBase.isdefined_logscale
MessagePassingRulesBase.getlogscale
```

## Declaring a log scale from the body

```@docs
MessagePassingRulesBase.with_logscale
MessagePassingRulesBase.from_body
```

## Reading the inputs' log scales

A rule that needs the log scales its inputs arrived with declares `reads_logscale = true` and
reads them as `args.logscale.m[:x]`, keyed like `args.m`, calling [`require_logscale`](@ref) on
those whose value it needs. Its caller must provide them: an engine does when it tracks log
scales (ReactiveMP's activation option `logscales = true`), and a call by hand takes them as
`logscale = (x = …,)`; without them the call is an error. The `Mixture` rules are such rules:
the message towards the switch is a softmax over the evidence of each component.

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
