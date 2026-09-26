# The log scale of a rule's result: the scalar that relates the function the result stands for
# to the normalised distribution the rule returns, `message(x) = exp(logscale) · result(x)`. A rule
# may stand for an unnormalised function for any reason; belief propagation, where
# `m(x) = ∫ f(x, y…) Π m_i(y_i)` is in general not normalised, is the common case. Where it can be
# computed, a rule declares it with the `logscale` keyword and an engine propagates it through
# products, as BayesBase's `compute_logscale` does for a product of two distributions; where it
# cannot, as for a naive variational message, it is undefined.

"""
    UndefinedLogScale(cause::Symbol, detail = nothing)

A log scale that is not known, with the reason: `cause` says why, `detail` what it concerns.
The causes an engine and the base package record:
- `:no_declaration`: the rule that computed the message declares no `logscale`; `detail` is its
  [`RuleSpec`](@ref);
- `:initial`: an initial message, not computed by a rule;
- `:fallback`: a message computed by a rule fallback;
- `:no_compute_logscale`: a product whose pair of distributions has no `compute_logscale`
  method; `detail` is the pair of types;
- `:form_constraint`: a product whose result a form constraint changed.

A cause outside this list is shown with its `detail`, so an engine may record its own.

It propagates: adding it to a number or to another undefined log scale gives an undefined log
scale, keeping the first reason. Nothing errors because a log scale is undefined, until
[`require_logscale`](@ref) is asked for its value. At the REPL it shows its reason.

```jldoctest
julia> u = UndefinedLogScale(:initial) + 1.5
UndefinedLogScale: the message is an initial one, not computed by a rule

julia> isdefined_logscale(u), isdefined_logscale(1.5)
(false, true)
```

See also [`isdefined_logscale`](@ref), [`UndefinedLogScaleError`](@ref).
"""
struct UndefinedLogScale
    cause::Symbol
    detail::Any
end

UndefinedLogScale(cause::Symbol) = UndefinedLogScale(cause, nothing)

Base.:+(a::UndefinedLogScale, ::Real) = a
Base.:+(::Real, b::UndefinedLogScale) = b
Base.:+(a::UndefinedLogScale, ::UndefinedLogScale) = a
Base.:+(a::UndefinedLogScale) = a

function describe_undefined(io::IO, logscale::UndefinedLogScale)
    cause, detail = logscale.cause, logscale.detail
    if cause === :no_declaration
        print(io, "the ", detail isa RuleSpec ? rule_heading(detail) : "rule", " declares no `logscale`")
    elseif cause === :initial
        print(io, "the message is an initial one, not computed by a rule")
    elseif cause === :fallback
        print(io, "the message was computed by a rule fallback")
    elseif cause === :no_compute_logscale
        print(io, "BayesBase has no `compute_logscale` for the product of ", detail === nothing ? "these distributions" : join(detail, " and "))
    elseif cause === :form_constraint
        print(io, "a form constraint changed the product")
    else
        print(io, cause, detail === nothing ? "" : ": $(detail)")
    end
    return nothing
end

function Base.show(io::IO, logscale::UndefinedLogScale)
    print(io, "UndefinedLogScale(")
    show(io, logscale.cause)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", logscale::UndefinedLogScale)
    print(io, "UndefinedLogScale: ")
    describe_undefined(io, logscale)
    return nothing
end

"""
    UndefinedLogScaleError(logscale::UndefinedLogScale)

The error [`require_logscale`](@ref) throws when a log scale a computation needs is not known.
Its message gives the reason the [`UndefinedLogScale`](@ref) records, such as the rule that
declares no `logscale`.
"""
struct UndefinedLogScaleError <: Exception
    logscale::UndefinedLogScale
end

function Base.showerror(io::IO, err::UndefinedLogScaleError)
    print(io, "UndefinedLogScaleError: a log scale is needed but not known: ")
    describe_undefined(io, err.logscale)
    return nothing
end

"""
    require_logscale(logscale) -> Real

Return `logscale` when it is a number, and throw when it is not known. A rule declared with
`reads_logscale = true` calls it on each incoming log scale whose value it needs.

# Throws
- [`UndefinedLogScaleError`](@ref), naming the reason, for an [`UndefinedLogScale`](@ref);
- `ArgumentError` for `nothing`: the caller does not track log scales.

```jldoctest
julia> require_logscale(-0.5)
-0.5

julia> require_logscale(UndefinedLogScale(:initial))
ERROR: UndefinedLogScaleError: a log scale is needed but not known: the message is an initial one, not computed by a rule
```
"""
require_logscale(logscale::Real) = logscale
require_logscale(logscale::UndefinedLogScale) = throw(UndefinedLogScaleError(logscale))
require_logscale(::Nothing) = throw(ArgumentError("log scales are not tracked; an engine tracks them when asked to, e.g. ReactiveMP's `logscales = true`"))

"""
    isdefined_logscale(logscale) -> Bool

Whether `logscale` is a number, rather than an [`UndefinedLogScale`](@ref) or `nothing`.
"""
isdefined_logscale(logscale) = logscale isa Real

"""
    WithLogScale(result, logscale)

A rule's result paired with its log scale, as [`with_logscale`](@ref) builds it. Only the body of
a rule declared with `logscale = from_body` returns one, and whoever runs the rule unwraps it; no
rule receives one, and a caller never sees one.
"""
struct WithLogScale{R, L}
    result::R
    logscale::L
end

"""
    with_logscale(result, logscale)
    with_logscale(; result, logscale)

Pair a rule's result with its log scale: what the body of a rule declared with
`logscale = from_body` returns, for a rule whose log scale shares its work with the result.
Running the rule unwraps it, so [`getresult`](@ref) is `result` and [`getlogscale`](@ref) is
`logscale`.

# Throws
Running the rule throws an `ArgumentError` when a body returns one without the rule declaring
`logscale = from_body`, or a rule declaring it returns anything else.

# Examples

```julia
@define_message_update_rule(
    node = Mixture, target = :out, algorithm = MixtureBP,
    args = (m[:switch]::Any, m[:inputs...]::Any),
    logscale = from_body, reads_logscale = true,
    body = (args) -> begin
        weights = ...
        with_logscale(MixtureDistribution(collect(args.m[:inputs]), softmax(weights)), logsumexp(weights))
    end,
)
```
"""
with_logscale(result, logscale) = WithLogScale(result, logscale)
with_logscale(; result, logscale) = WithLogScale(result, logscale)

"""
    FromBody

The type of [`from_body`](@ref), the marker a rule's `logscale` declaration is compared against.
"""
struct FromBody end

"""
    from_body

The `logscale` declaration of a rule whose body computes its log scale: written
`logscale = from_body`, the body then returns [`with_logscale`](@ref)`(result, logscale)`.
"""
const from_body = FromBody()

"""
    RuleLogScales(; m = NamedTuple())

The log scales that arrived with a rule's inbound messages, `args.logscale` of a
[`RuleArgs`](@ref), read as `args.logscale.m[:out]` and keyed exactly like `args.m`; a group is a
tuple under its name. Each is a number or an [`UndefinedLogScale`](@ref). A rule reads them only
when declared with `reads_logscale = true`, and calls [`require_logscale`](@ref) on those whose
value it needs.
"""
struct RuleLogScales{M <: Messages}
    m::M
end

RuleLogScales(; m = NamedTuple()) = RuleLogScales(as_messages(m))

"""
    getlogscale(x)

The log scale of `x`: a rule's [`RuleResult`](@ref), or, in an engine that adds methods, a
message or a marginal. A number, an [`UndefinedLogScale`](@ref) with its reason, or `nothing`
where log scales are not tracked. For a `RuleResult` it is what the rule declares for a message,
and `nothing` for a marginal or an average energy.
"""
function getlogscale end

# A `logscale` declaration is a number, a function of the rule's inputs, `from_body`, or
# `nothing`: the rule declares none.
valid_logscale_declaration(declaration) = declaration === nothing || declaration isa Union{Real, Function, FromBody}

describe_logscale_declaration(::Nothing) = "none"
describe_logscale_declaration(::FromBody) = "from the body"
describe_logscale_declaration(::Function) = "a function of the inputs"
describe_logscale_declaration(declaration::Real) = string(declaration)

# The log scale a rule declares, for a call with these arguments and this raw result.
@inline rule_logscale(spec, ::Nothing, raw, algorithm, ctx, args, target) = UndefinedLogScale(:no_declaration, spec)
@inline rule_logscale(spec, declaration::Real, raw, algorithm, ctx, args, target) = declaration
@inline rule_logscale(spec, declaration::FromBody, raw, algorithm, ctx, args, target) = raw.logscale
@inline rule_logscale(spec, declaration::Function, raw, algorithm, ctx, args, target) = declaration(algorithm, ctx, args, target)

# A rule declared `from_body` returns a `WithLogScale`; any other returns its result bare.
@inline unwrap_result(spec, raw::WithLogScale) =
    spec.logscale isa FromBody ? raw.result : throw(ArgumentError("the $(rule_heading(spec)) returned `with_logscale(...)` but does not declare `logscale = from_body`"))
@inline unwrap_result(spec, raw) =
    spec.logscale isa FromBody ? throw(ArgumentError("the $(rule_heading(spec)) declares `logscale = from_body` but its body did not return `with_logscale(result, logscale)`")) : raw
as_logscales(logscale::RuleLogScales) = logscale
