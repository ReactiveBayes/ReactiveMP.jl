# The context a node's rules run with: the engine's services, merged with those the node was
# activated with (`FactorNodeActivationOptions`'s `context`), which add to them or override them.
# One `RuleContext` per node, built at activation and shared by its mappings.

"""
    ReactiveMP.node_context(factornode, context = NamedTuple())

The [`MessagePassingRulesBase.RuleContext`](@ref) the rules of `factornode` run with: `node`, the
node itself; `rng`, the task's own random number generator; `matrix_correction`, `nothing`, so
each rule applies its own; merged with `context`, a `NamedTuple` of services, or a
`RuleContext`, which adds services a rule needs or overrides these.
"""
node_context(factornode, context::NamedTuple = NamedTuple()) =
    RuleContext(merge((node = factornode, rng = Random.default_rng(), matrix_correction = nothing), context))
node_context(factornode, context::RuleContext) = node_context(factornode, getfield(context, :services))
node_context(factornode, ::Nothing) = node_context(factornode)
