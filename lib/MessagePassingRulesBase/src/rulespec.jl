"""
    RuleSpec

A rule as data, and the thing that runs. Dispatch resolves a call to a `RuleSpec`, which
holds the body and the preallocation function and knows whether it is in-place. It has no
type parameters, so resolution returns one concrete type wherever it can be inferred.

`kind` is `:message`, `:marginal` or `:average_energy`. The body takes the full slot list
`(output, algo, ctx, args, ann, target)`; `prealloc`, when present, takes
`(algo, ctx, args)`.
"""
struct RuleSpec
    kind::Symbol
    node::Any
    target::Any
    algorithm::Type
    signature::Type
    body::Function
    prealloc::Union{Nothing, Function}
    inplace::Bool
    pure::Bool
    services::Tuple{Vararg{Symbol}}
    source::String
    file::Symbol
    line::Int
end

function RuleSpec(;
        kind::Symbol, node, target, algorithm::Type, signature::Type, body::Function,
        prealloc = nothing, inplace::Bool = false, pure::Union{Nothing, Bool} = nothing,
        services::Tuple{Vararg{Symbol}} = (), source::AbstractString = "",
        file::Symbol = :none, line::Integer = 0,
    )
    kind in (:message, :marginal, :average_energy) ||
        throw(ArgumentError("rule kind must be :message, :marginal or :average_energy, got :$kind"))
    inplace && prealloc === nothing &&
        throw(ArgumentError("an in-place rule needs a `preallocate` function"))
    for service in services
        service in CONTEXT_SERVICES ||
            throw(ArgumentError("unknown context service :$service; valid services are $CONTEXT_SERVICES"))
    end
    effective = something(pure, algorithm <: AbstractAlgorithm ? ispure(algorithm) : true)
    return RuleSpec(
        kind, node, target, algorithm, signature, body, prealloc, inplace, effective,
        services, String(source), file, Int(line),
    )
end

"""
    RuleNotFound

What resolution returns when no rule matches. Resolution never throws.
"""
struct RuleNotFound
    kind::Symbol
    node::Any
    target::Any
    algorithm::Any
    args::Any
end

"""
    find_message_rule(node, target, algorithm, args)

The [`RuleSpec`](@ref) for a message towards `target`, or a [`RuleNotFound`](@ref). Rule
definitions add methods to this function; it never runs a rule and never throws.
"""
find_message_rule(node, target, algorithm, args) = RuleNotFound(:message, node, target, algorithm, args)

"""
    find_marginal_rule(node, cluster, algorithm, args)

As [`find_message_rule`](@ref), for the marginal of a structural cluster.
"""
find_marginal_rule(node, cluster, algorithm, args) = RuleNotFound(:marginal, node, cluster, algorithm, args)

"""
    find_average_energy(node, algorithm, args)

As [`find_message_rule`](@ref), for a node's average energy.
"""
find_average_energy(node, algorithm, args) = RuleNotFound(:average_energy, node, nothing, algorithm, args)

"""
    RuleNotFoundError(notfound::RuleNotFound)
"""
struct RuleNotFoundError <: Exception
    notfound::RuleNotFound
end

function Base.showerror(io::IO, err::RuleNotFoundError)
    nf = err.notfound
    print(io, "no ", nf.kind, " rule found for ", nf.node)
    nf.target === nothing || print(io, " towards ", nf.target)
    print(io, " under ", nf.algorithm)
    return nothing
end

"""
    execute_rule(spec, output, algorithm, ctx, args, ann, target)

Run a resolved rule. For an in-place rule, `output` is the buffer to write into; `nothing`
asks the rule to preallocate one. Nothing here catches exceptions: whatever a rule throws
propagates to the caller.
"""
@inline function execute_rule(spec::RuleSpec, output, algorithm, ctx, args, ann, target)
    if spec.inplace && output === nothing
        output = spec.prealloc(algorithm, ctx, args)
    end
    return spec.body(output, algorithm, ctx, args, ann, target)
end

@inline function resolved(spec)
    spec isa RuleNotFound && throw(RuleNotFoundError(spec))
    return spec
end

"""
    message_passing_rule(node, target, algorithm, args[, ctx, ann])

Resolve the message rule towards `target` and run it, allocating its result.
"""
@inline function message_passing_rule(node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = resolved(find_message_rule(node, target, algorithm, args))
    return execute_rule(spec, nothing, algorithm, ctx, args, ann, target)
end

"""
    message_passing_rule!(output, node, target, algorithm, args[, ctx, ann])

Resolve an in-place message rule and run it into `output`.
"""
@inline function message_passing_rule!(output, node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = resolved(find_message_rule(node, target, algorithm, args))
    spec.inplace || throw(ArgumentError("the rule for $node towards $target has no in-place form"))
    return execute_rule(spec, output, algorithm, ctx, args, ann, target)
end

"""
    message_passing_marginalrule(node, cluster, algorithm, args[, ctx, ann])

Resolve the marginal rule for `cluster` and run it.
"""
@inline function message_passing_marginalrule(node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = resolved(find_marginal_rule(node, cluster, algorithm, args))
    return execute_rule(spec, nothing, algorithm, ctx, args, ann, cluster)
end

"""
    message_passing_marginalrule!(output, node, cluster, algorithm, args[, ctx, ann])
"""
@inline function message_passing_marginalrule!(output, node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = resolved(find_marginal_rule(node, cluster, algorithm, args))
    spec.inplace || throw(ArgumentError("the marginal rule for $node over $cluster has no in-place form"))
    return execute_rule(spec, output, algorithm, ctx, args, ann, cluster)
end

"""
    message_passing_average_energy(node, algorithm, args[, ctx, ann])

Resolve and compute a node's average energy.
"""
@inline function message_passing_average_energy(node, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = resolved(find_average_energy(node, algorithm, args))
    return execute_rule(spec, nothing, algorithm, ctx, args, ann, nothing)
end

"""
    missing_services(spec::RuleSpec, ctx::RuleContext)

The context services a rule declares and `ctx` does not provide. Meant to be checked once,
when a node is set up, rather than on every call.
"""
missing_services(spec::RuleSpec, ctx::RuleContext) =
    filter(service -> getfield(ctx, service) === nothing, spec.services)
