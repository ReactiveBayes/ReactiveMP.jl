"""
    RuleContext(; services...)
    RuleContext(services::NamedTuple)

The context a rule receives as `ctx`: the services its caller supplies, read as `ctx.name`. It
never takes part in dispatch. A rule declares the services it reads, `ctx = (:rng,)`; any name
is allowed, so a rule may need a service of its own. A service the caller does not supply reads
as `nothing`, and [`missing_services`](@ref) lists them; an engine refuses to run such a rule,
through [`check_services`](@ref), while an interactive call runs it with what it is given.

The services an engine supplies by default, [`DEFAULT_CONTEXT_SERVICES`](@ref):
- `node`: the factor node the rule belongs to, e.g. for [`getnodefn`](@ref)`(ctx.node, target)`;
- `rng`: the random number generator, owned by the caller;
- `matrix_correction`: how a rule corrects a matrix it builds before using it, such as a
  precision that must stay positive definite: a strategy from MatrixCorrectionTools, or
  `nothing`, and each rule then applies its own default; a rule reads it through
  [`matrix_correction`](@ref)`(ctx, default)`. MatrixCorrectionTools' `NoCorrection()` is an
  explicit identity.

A mutable object holding the services as a typed `NamedTuple`: a caller builds one and passes
it to every call by reference, and reading a service is inferred. `propertynames(ctx)` gives
the services it supplies, and `merge(ctx, services::NamedTuple)` returns a context with
services added or overridden, as a caller layers its own over the defaults.

```jldoctest
julia> using MessagePassingRulesBase: RuleContext

julia> ctx = merge(RuleContext(rng = 1), (scale = 2.0,));

julia> ctx.rng, ctx.scale, ctx.node
(1, 2.0, nothing)

julia> propertynames(ctx)
(:rng, :scale)
```

See also [`check_services`](@ref), [`matrix_correction`](@ref).
"""
mutable struct RuleContext{S <: NamedTuple}
    const services::S
end

RuleContext(; services...) = RuleContext(NamedTuple(services))

Base.getproperty(ctx::RuleContext, name::Symbol) = get(getfield(ctx, :services), name, nothing)
Base.propertynames(ctx::RuleContext) = keys(getfield(ctx, :services))
Base.merge(ctx::RuleContext, services::NamedTuple) = RuleContext(merge(getfield(ctx, :services), services))
Base.show(io::IO, ctx::RuleContext) = print(io, "RuleContext", getfield(ctx, :services))

"""
    DEFAULT_CONTEXT_SERVICES

The context services an engine supplies by default: `node`, `rng` and `matrix_correction` (see
[`RuleContext`](@ref)). A rule may declare others.
"""
const DEFAULT_CONTEXT_SERVICES = (:node, :rng, :matrix_correction)

"""
    matrix_correction(ctx::RuleContext, default)

The matrix correction a rule applies: `ctx.matrix_correction` when it is set, and the rule's
own `default` when it is `nothing` or not supplied. A rule that corrects nothing by default
passes `nothing`. The rule declares `ctx = (:matrix_correction,)`, and applies the strategy with
MatrixCorrectionTools' `correction!`.

```julia
W = correction!(matrix_correction(ctx, ReplaceZeroDiagonalEntries(tiny)), A * w * A')
```
"""
matrix_correction(ctx::RuleContext, default) = ctx.matrix_correction === nothing ? default : ctx.matrix_correction
