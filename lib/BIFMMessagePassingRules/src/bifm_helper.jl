"""
    BIFMHelper

The node that starts a chain of [`BIFM`](@ref) nodes, `out ~ BIFMHelper(in)`, with `out` the
first state and `in` its prior. It turns the backward pass into the forward one: towards `in` it
passes the backward message on `out` through, and towards `out` it sends the marginal of `in` as a
`TerminalProdArgument`, the forward pass's starting marginal.

Its rule towards `in` reads the message on `out`, and its rule towards `out` the marginal of
`in`: the model keeps `in` and `out` in separate clusters.
"""
struct BIFMHelper end

@define_factor_node(
    node = BIFMHelper, type = Stochastic, interfaces = [:out, :in],
    dependencies = [:in => (m[:out],), :out => (default,)],
)

@define_message_update_rule(node = BIFMHelper, target = :in, args = (m[:out]::Any,), body = (args) -> args.m[:out])

@define_message_update_rule(node = BIFMHelper, target = :out, args = (q[:in]::Any,), body = (args) -> TerminalProdArgument(args.q[:in]))

# The free energy of a BIFM model is not supported.
@define_average_energy(node = BIFMHelper, args = (q[:out]::Any, q[:in]::Any), body = (args) -> throw(BIFMFreeEnergyError(:BIFMHelper)))
