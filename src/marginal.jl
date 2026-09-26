export Marginal, getdata, is_clamped, is_initial, as_marginal

using Distributions
using Rocket

import Rocket: getrecent
import Base: ==, ndims, precision, length, size, iterate

# The representation is a mutable struct with `const` fields: measured faster than an
# immutable one through the equality chain, and lighter everywhere
# (`scripts/benchmark_message_representation.jl`).
"""
    Marginal(data, is_clamped::Bool, is_initial::Bool)
    Marginal(data, is_clamped::Bool, is_initial::Bool, annotations::AnnotationDict)
    Marginal(data, is_clamped::Bool, is_initial::Bool, annotations::AnnotationDict, logscale)

A marginal, the belief about a variable or a cluster of a factor node's variables: its data,
usually a distribution, with the flags and the metadata the engine tracks for it.

# Arguments

- `data`: the marginal itself, usually a distribution; the joint of a cluster may be a
  [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster), and a marginal with a
  `missing` input is `missing`;
- `is_clamped`: whether the marginal comes from constants and observations alone. $(DOC_CLAMPED)
- `is_initial`: whether the marginal was set before inference or computed from initial values
  only. $(DOC_INITIAL)
- `annotations`: the [`ReactiveMP.AnnotationDict`](@ref) of optional metadata, empty by default;
- `logscale`: the log scale of the product of messages the marginal was formed from (see
  [`getlogscale`](@ref)), or `nothing`, the default.

$(DOC_STATISTICS)

$(DOC_EQUALITY)

# Examples

```jldoctest
julia> marginal = Marginal(Gamma(10.0, 2.0), false, true)
Marginal(Distributions.Gamma{Float64}(α=10.0, θ=2.0))

julia> mean(marginal)
20.0

julia> is_clamped(marginal), is_initial(marginal)
(false, true)
```

See also [`Message`](@ref), [`as_marginal`](@ref).
"""
mutable struct Marginal{D, L}
    const data::D
    const is_clamped::Bool
    const is_initial::Bool
    const annotations::AnnotationDict
    const logscale::L
end

Marginal(data, is_clamped::Bool, is_initial::Bool) =
    Marginal(data, is_clamped, is_initial, AnnotationDict(), nothing)
Marginal(data, is_clamped::Bool, is_initial::Bool, annotations::AnnotationDict) =
    Marginal(data, is_clamped, is_initial, annotations, nothing)

function Base.show(io::IO, marginal::Marginal)
    print(io, "Marginal(", getdata(marginal), ")")
    marginal.logscale === nothing || print(io, " with logscale = ", marginal.logscale)
    ann = getannotations(marginal)
    return if !isempty(ann)
        print(io, " with ", ann)
    end
end

function Base.:(==)(left::Marginal, right::Marginal)
    # We need this dummy method as Julia is not smart enough to
    # do that automatically if `data` is mutable
    return left.is_clamped == right.is_clamped &&
        left.is_initial == right.is_initial &&
        left.data == right.data
end

"""
    getdata(marginal::Marginal)

The data of `marginal`, usually a distribution, or `missing`.
"""
getdata(marginal::Marginal) = marginal.data

"""
    is_clamped(marginal::Marginal) -> Bool

Whether `marginal` comes from constants and observations alone. $(DOC_CLAMPED)
"""
is_clamped(marginal::Marginal) = marginal.is_clamped

"""
    is_initial(marginal::Marginal) -> Bool

Whether `marginal` was set before inference or computed from initial values only. $(DOC_INITIAL)
"""
is_initial(marginal::Marginal) = marginal.is_initial

"""
    getannotations(marginal::Marginal) -> AnnotationDict

The [`ReactiveMP.AnnotationDict`](@ref) of `marginal`: a variable's marginal keeps the annotations
of the product it was formed from; the joint marginal of a cluster has none.
"""
getannotations(marginal::Marginal) = marginal.annotations

"""
    getlogscale(marginal::Marginal)

The log scale of `marginal`: that of the product of the messages a random variable formed it
from, a number or an [`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale)
saying why it is not known. For a variable's marginal in a tree-shaped model inferred exactly by
belief propagation, it is the log evidence of the data.

A data variable's marginal, an observation, has log scale zero, and so does a constant's.

# Throws

- `ArgumentError` where the marginal carries no log scale: when log scales are not tracked, and
  always for an initial marginal set with [`ReactiveMP.set_initial_marginal!`](@ref) and for the
  joint marginal of a factor node's cluster, which carry none even with `logscales = true`.
"""
getlogscale(marginal::Marginal) = marginal.logscale === nothing ? throw(
        ArgumentError(
            "this marginal carries no log scale: either log scales are not tracked (activate the graph with `logscales = true`), " *
            "or it is an initial marginal, set with `set_initial_marginal!`, or the joint marginal of a node's cluster, which carry none",
        ),
    ) : marginal.logscale

typeofdata(marginal::Marginal) = typeof(getdata(marginal))

getdata(marginals::NTuple{N, <:Marginal}) where {N} = map(getdata, marginals)
getdata(marginals::AbstractArray{<:Marginal}) = map(getdata, marginals)

ispointmass(marginal::Marginal) = ispointmass(marginal, getdata(marginal))
ispointmass(marginal::Marginal, data::PointMass) = true
ispointmass(marginal::Marginal, data) = false

## Statistics

Distributions.pdf(marginal::Marginal, x) = Distributions.pdf(getdata(marginal), x)
Distributions.logpdf(marginal::Marginal, x) = Distributions.logpdf(getdata(marginal), x)

MacroHelpers.@proxy_methods Marginal getdata [
    BayesBase.mean,
    BayesBase.median,
    BayesBase.mode,
    BayesBase.shape,
    BayesBase.scale,
    BayesBase.rate,
    BayesBase.var,
    BayesBase.std,
    BayesBase.cov,
    BayesBase.invcov,
    BayesBase.logdetcov,
    BayesBase.entropy,
    BayesBase.params,
    BayesBase.mean_cov,
    BayesBase.mean_var,
    BayesBase.mean_invcov,
    BayesBase.mean_precision,
    BayesBase.weightedmean_cov,
    BayesBase.weightedmean_var,
    BayesBase.weightedmean_invcov,
    BayesBase.weightedmean_precision,
    BayesBase.probvec,
    BayesBase.weightedmean,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
]

# Eltype is special here, because it should be only defined on types
# Otherwise it causes invalidations and slower compile times
Base.eltype(::Type{<:Marginal{D}}) where {D} = Base.eltype(D)

Distributions.mean(fn::Function, marginal::Marginal) =
    mean(fn, getdata(marginal))

"""
    as_marginal(message::Message) -> Marginal
    as_marginal(marginal::Marginal) -> Marginal

A message as a [`Marginal`](@ref), with its flags, annotations and log scale, its data in its public
type: an efficient working type such as `WishartFast` becomes the type users expect
([`public_equivalent`](@extref MessagePassingRulesBase.public_equivalent)). A marginal is returned
as it is. A variable forms its marginal from the product of its messages with it.
"""
function as_marginal end

as_marginal(marginal::Marginal) = marginal

skip_initial() = filter(v -> !is_initial(v))

## Marginal observable

"""
    MarginalObservable()

The stream of a variable's marginals, or of a factor node's joint marginal of a cluster. Every
subscriber shares one upstream subscription, and the latest marginal is kept, so
`Rocket.getrecent` returns it and a late subscriber receives it at once.

The stream is lazy: activation connects it to its source. Before that,
[`ReactiveMP.set_initial_marginal!`](@ref) can seed it with an initial marginal, which is what a
rule reads before any marginal has been computed.

See also [`ReactiveMP.MessageObservable`](@ref), [`ReactiveMP.get_stream_of_marginals`](@ref).
"""
struct MarginalObservable <: Subscribable{Marginal}
    subject::Rocket.RecentSubjectInstance{Marginal, Subject{Marginal, AsapScheduler, AsapScheduler}}
    stream::LazyObservable{Marginal}
end

MarginalObservable() =
    MarginalObservable(RecentSubject(Marginal), lazy(Marginal))

Rocket.getrecent(observable::MarginalObservable) =
    Rocket.getrecent(observable.subject)

@inline Rocket.on_subscribe!(observable::MarginalObservable, actor) =
    subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Actor{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.NextActor{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ErrorActor{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.CompletionActor{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Subject{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.BehaviorSubjectInstance{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.PendingSubjectInstance{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.RecentSubjectInstance{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ReplaySubjectInstance{<:Marginal}) = Rocket.on_subscribe!(observable.stream, actor)

function connect!(marginal::MarginalObservable, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function set_initial_marginal!(marginal::MarginalObservable, value)
    next!(marginal.subject, Marginal(value, false, true))
    return nothing
end

## Marginal Mapping structure
## https://github.com/JuliaLang/julia/issues/42559
## Explanation: Julia cannot fully infer type of the lambda callback function in activate! method in node.jl file
## We create a lambda-like callable structure to improve type inference and make it more stable
"""
    MarginalMapping

What computes the joint marginal of one cluster of a factor node, called with the latest
messages and marginals the node's marginal rule depends on. A node builds one for each joint
cluster at activation.

A call returns a `missing` marginal, and runs no rule, when an input is `missing`. Otherwise it
resolves the marginal rule with
[`find_marginal_rule`](@extref MessagePassingRulesBase.find_marginal_rule) under the node's
algorithm, checks it against the [`ReactiveMP.EngineDiagnostics`](@ref) and the services the node's
context supplies ([`check_services`](@extref MessagePassingRulesBase.check_services)), and runs it.
Where no rule matches, it throws a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError): a rule fallback does
not apply to a marginal.

The marginal is clamped when every input is, and initial when it is not clamped and every input
is clamped or initial. It carries no annotations and no log scale, and no callback is invoked.
"""
struct MarginalMapping{F, T, N, M, A, R, G}
    target::T
    msgs_names::N
    marginals_names::M
    algorithm::A
    factornode::R
    diagnostics::EngineDiagnostics
    context::G
    scratch::ScratchSlot
end

marginal_mapping_fform(::MarginalMapping{F}) where {F} = F
marginal_mapping_fform(::MarginalMapping{F}) where {F <: Function} = F.instance

MarginalMapping(::Type{F}, target::T, msgs_names::N, marginals_names::M, algorithm::A, factornode::R, diagnostics::EngineDiagnostics = EngineDiagnostics(), context = nothing) where {F, T, N, M, A, R} =
    (c = node_context(factornode, context); MarginalMapping{F, T, N, M, A, R, typeof(c)}(target, msgs_names, marginals_names, algorithm, factornode, diagnostics, c, ScratchSlot()))

MarginalMapping(::F, target::T, msgs_names::N, marginals_names::M, algorithm::A, factornode::R, diagnostics::EngineDiagnostics = EngineDiagnostics(), context = nothing) where {F <: Function, T, N, M, A, R} =
    (c = node_context(factornode, context); MarginalMapping{F, T, N, M, A, R, typeof(c)}(target, msgs_names, marginals_names, algorithm, factornode, diagnostics, c, ScratchSlot()))

function (mapping::MarginalMapping)(dependencies)
    messages = getrecent(dependencies[1])
    marginals = getrecent(dependencies[2])

    # Marginal is clamped if all of the inputs are clamped
    is_marginal_clamped =
        __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Marginal is initial if it is not clamped and all of the inputs are either clamped or initial
    is_marginal_initial =
        !is_marginal_clamped && (
        __check_all(is_clamped_or_initial, messages) &&
            __check_all(is_clamped_or_initial, marginals)
    )

    marginal = if has_missing_inputs(messages) || has_missing_inputs(marginals)
        missing
    else
        compute_marginal(mapping, messages, marginals)
    end

    return Marginal(marginal, is_marginal_clamped, is_marginal_initial)
end

function compute_marginal(mapping::MarginalMapping, messages, marginals)
    fform = marginal_mapping_fform(mapping)
    args = rule_arguments(mapping.msgs_names, messages, mapping.marginals_names, marginals)
    spec = audit_rule(mapping.diagnostics, resolve_rule(MessagePassingRulesBase.find_marginal_rule(fform, mapping.target, mapping.algorithm, args)))
    ctx = mapping.context
    MessagePassingRulesBase.check_services(spec, ctx)
    ann = rule_annotations(mapping.msgs_names, messages, mapping.marginals_names, marginals, MessagePassingRulesBase.NoAnnotations())
    algorithm = MessagePassingRulesBase.rule_algorithm(spec, mapping.algorithm)
    scratch = scratch_for!(mapping.scratch, spec, algorithm, ctx, args, mapping.target, mapping.diagnostics.checked_buffers)
    return MessagePassingRulesBase.execute_rule(spec, nothing, scratch, algorithm, ctx, args, ann, mapping.target)
end

Base.map(::Type{T}, mapping::M) where {T, M <: MarginalMapping} =
    Rocket.MapOperator{T, M}(mapping)
