"""
    BIFMHelper

The node that starts a chain of [`BIFM`](@ref) nodes, `out ~ BIFMHelper(in)`, with `out` the
first state and `in` its prior. It turns the backward pass into the forward one: towards `in` it
passes the backward message on `out` through, and towards `out` it sends the marginal of `in` as a
`TerminalProdArgument`, the forward pass's starting marginal.

Its rule towards `in` reads the message on `out`, and its rule towards `out` the marginal of
`in`, as v6 declared: the model keeps `in` and `out` in separate clusters.
"""
struct BIFMHelper end

@define_factor_node(
    node = BIFMHelper, type = Stochastic, interfaces = [:out, :in],
    dependencies = [:in => (m[:out],), :out => (default,)],
)

@define_message_update_rule(node = BIFMHelper, target = :in, args = (m[:out]::Any,), body = (args) -> args.m[:out])

@define_message_update_rule(node = BIFMHelper, target = :out, args = (q[:in]::Any,), body = (args) -> TerminalProdArgument(args.q[:in]))

# v6's energy was the entropy of q(in), a trick to cancel its term in a free energy that, for a
# BIFM model, could not be computed anyway.
@define_average_energy(node = BIFMHelper, args = (q[:out]::Any, q[:in]::Any), body = (args) -> throw(BIFMFreeEnergyError(:BIFMHelper)))
