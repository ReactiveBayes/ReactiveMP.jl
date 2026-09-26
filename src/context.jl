# The context a node's rules run with: the engine's services, merged with those the node was
# activated with (`FactorNodeActivationOptions`'s `context`), which add to them or override them.
# One `RuleContext` per node, built at activation and shared by its mappings.

"""
    ReactiveMP.node_context(factornode, context = NamedTuple()) -> RuleContext

The [`RuleContext`](@extref MessagePassingRulesBase.RuleContext) the rules of `factornode` run
with, read by a rule as `ctx`. The engine supplies three services:

- `node`: the factor node itself, which a rule reaches its function through
  ([`getnodefn`](@extref MessagePassingRulesBase.getnodefn));
- `rng`: the task's own random number generator, `Random.default_rng()`;
- `matrix_correction`: `nothing`, so each rule applies its own.

`context` is merged over them: a `NamedTuple` of services, a `RuleContext`, or `nothing` for
none. It adds the services a rule declares and the engine does not supply, and overrides the
engine's. Activation builds the context once per node, from the activation option `context` (see
[`ReactiveMP.FactorNodeActivationOptions`](@ref)), and its message and marginal rules share it.

# Examples

```jldoctest
julia> ctx = ReactiveMP.node_context(nothing, (scale = 2.0,));

julia> ctx.scale, ctx.matrix_correction
(2.0, nothing)
```
"""
node_context(factornode, context::NamedTuple = NamedTuple()) =
    RuleContext(merge((node = factornode, rng = Random.default_rng(), matrix_correction = nothing), context))
node_context(factornode, context::RuleContext) = node_context(factornode, getfield(context, :services))
node_context(factornode, ::Nothing) = node_context(factornode)
