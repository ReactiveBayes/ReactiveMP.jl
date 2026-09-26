"""
    BIFMHelper

The stochastic node that starts a chain of [`BIFM`](@ref) nodes, `out ~ BIFMHelper(in)`, with
`out` the first state and `in` its prior. It turns the backward pass into the forward one:
towards `in` it passes the backward message on `out` through, and towards `out` it sends the
marginal of `in` as a `TerminalProdArgument`, the starting marginal of the forward pass.

# Interfaces

- `out`: the first state of the chain;
- `in`: its prior.

Its rule towards `in` reads the message on `out`, and its rule towards `out` the marginal of
`in`, of any type: the model keeps `in` and `out` in separate clusters, `q(in) q(out)`. It runs
under the default algorithm, and needs none.

# Limitations

- **No free energy.** Its average energy throws a
  [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).

# Examples

```jldoctest; setup = :(using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = BIFMHelper, target = :out,
           q = (in = MvNormalMeanCovariance([1.0], [2.0;;]),),
       );

julia> getresult(result) isa TerminalProdArgument
true
```

See also [`BIFM`](@ref), [`BIFMSmoother`](@ref).
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
