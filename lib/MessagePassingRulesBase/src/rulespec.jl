"""
    InputSpec

One declared input of a rule: container `:m` or `:q`, the interface or cluster `key`, and
the `selection` — `:single`, `:cluster`, or for a group `:all`, `:aligned` or `:allbutself`.
"""
struct InputSpec
    container::Symbol
    key::Union{Symbol, Tuple{Vararg{ClusterMember}}}
    selection::Symbol
    type::Any
end

"""
    RuleSpec

A rule as data, and the thing that runs. Dispatch resolves a call to a `RuleSpec`, which
holds the body and the preallocation function and knows whether it is in-place. It has no
type parameters, so resolution returns one concrete type wherever it can be inferred.

`default` marks a rule declared with `default` among its `args`, which takes whatever inputs the
factorisation delivers; `inputs` are then its typed ones only.

`kind` is `:message`, `:marginal` or `:average_energy`. The body takes the full slot list
`(output, scratch, algo, ctx, args, ann, target)`; `prealloc` and `scratch`, when present,
take `(algo, ctx, args, target)`.
"""
struct RuleSpec
    kind::Symbol
    node::Any
    target::Any
    algorithm::Type
    signature::Type
    inputs::Tuple{Vararg{InputSpec}}
    body::Function
    prealloc::Union{Nothing, Function}
    scratch::Union{Nothing, Function}
    default::Bool
    inplace::Bool
    pure::Bool
    services::Tuple{Vararg{Symbol}}
    source::String
    file::Symbol
    line::Int
end

function RuleSpec(;
        kind::Symbol, node, target, algorithm::Type, signature::Type, body::Function,
        inputs::Tuple{Vararg{InputSpec}} = (),
        prealloc = nothing, scratch = nothing, default::Bool = false, inplace::Bool = false, pure::Union{Nothing, Bool} = nothing,
        services::Tuple{Vararg{Symbol}} = (), source::AbstractString = "",
        file::Symbol = :none, line::Integer = 0,
    )
    kind in (:message, :marginal, :average_energy) ||
        throw(ArgumentError("rule kind must be :message, :marginal or :average_energy, got :$kind"))
    inplace && prealloc === nothing &&
        throw(ArgumentError("an in-place rule needs a `preallocate` function"))
    effective = something(pure, algorithm <: AbstractAlgorithm ? ispure(algorithm) : true)
    return RuleSpec(
        kind, node, target, algorithm, signature, inputs, body, prealloc, scratch, default, inplace, effective,
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
definitions add methods to this function; it never runs a rule and never throws. For a
[`DefaultAlgorithmExtension`](@ref) without a rule of its own, the default's rule is returned;
a `RuleNotFound` always names the algorithm the call asked for.
"""
function find_message_rule(node, target, algorithm, args)
    algorithm isa DefaultAlgorithmExtension || return RuleNotFound(:message, node, target, algorithm, args)
    inherited = find_message_rule(node, target, DefaultAlgorithm(), args)
    return inherited isa RuleSpec ? inherited : RuleNotFound(:message, node, target, algorithm, args)
end

"""
    find_marginal_rule(node, cluster, algorithm, args)

As [`find_message_rule`](@ref), for the marginal of a structural cluster.
"""
function find_marginal_rule(node, cluster, algorithm, args)
    algorithm isa DefaultAlgorithmExtension || return RuleNotFound(:marginal, node, cluster, algorithm, args)
    inherited = find_marginal_rule(node, cluster, DefaultAlgorithm(), args)
    return inherited isa RuleSpec ? inherited : RuleNotFound(:marginal, node, cluster, algorithm, args)
end

"""
    find_average_energy(node, algorithm, args)

As [`find_message_rule`](@ref), for a node's average energy.
"""
function find_average_energy(node, algorithm, args)
    algorithm isa DefaultAlgorithmExtension || return RuleNotFound(:average_energy, node, nothing, algorithm, args)
    inherited = find_average_energy(node, DefaultAlgorithm(), args)
    return inherited isa RuleSpec ? inherited : RuleNotFound(:average_energy, node, nothing, algorithm, args)
end

"""
    rule_algorithm(spec::RuleSpec, algorithm)

The algorithm value to run `spec` with, for a call made under `algorithm`: the call's own, or
`DefaultAlgorithm()` when `spec` was reached through an extension's fallback to the default.
An engine calls it before [`execute_rule`](@ref), as the `message_passing_*` functions do.
"""
@inline rule_algorithm(spec::RuleSpec, algorithm) = algorithm isa spec.algorithm ? algorithm : DefaultAlgorithm()

"""
    RuleNotFoundError(notfound::RuleNotFound)
"""
struct RuleNotFoundError <: Exception
    notfound::RuleNotFound
end

"""
    execute_rule(spec, output, algorithm, ctx, args, ann, target)
    execute_rule(spec, output, scratch, algorithm, ctx, args, ann, target)

Run a resolved rule. For an in-place rule, `output` is the buffer to write into; `nothing`
asks the rule to preallocate one. `scratch` is the rule's working memory, which an engine
builds once with [`rule_scratch`](@ref) and passes on every call; without it, or with
`nothing`, a rule that declares scratch gets a fresh one. Nothing here catches exceptions:
whatever a rule throws propagates to the caller.

A rule never sees a missing input. When any input is `missing`, an engine does not call the
rule at all, and does not run the annotation processors that follow a rule either; the
result is `missing`, carrying only the annotations written before the call.
"""
@inline execute_rule(spec::RuleSpec, output, algorithm, ctx, args, ann, target) =
    execute_rule(spec, output, nothing, algorithm, ctx, args, ann, target)

@inline function execute_rule(spec::RuleSpec, output, scratch, algorithm, ctx, args, ann, target)
    if spec.inplace && output === nothing
        output = spec.prealloc(algorithm, ctx, args, target)
    end
    if scratch === nothing
        scratch = rule_scratch(spec, algorithm, ctx, args, target)
    end
    return spec.body(output, scratch, algorithm, ctx, args, ann, target)
end

"""
    rule_scratch(spec, algorithm, ctx, args, target)

The working memory a rule declares with `scratch`, built from these inputs, or `nothing` for a
rule that declares none. An engine builds it once per outbound stream and passes it to every
[`execute_rule`](@ref) of that rule. The rule writes it before reading it, so the engine may
rebuild it whenever it likes.
"""
@inline rule_scratch(spec::RuleSpec, algorithm, ctx, args, target) =
    spec.scratch === nothing ? nothing : spec.scratch(algorithm, ctx, args, target)

@inline function throw_if_not_found(spec)
    spec isa RuleNotFound && throw(RuleNotFoundError(spec))
    return spec
end

"""
    message_passing_rule(node, target, algorithm, args[, ctx, ann])

Resolve the message rule towards `target` and run it, allocating its result.
"""
@inline function message_passing_rule(node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_message_rule(node, target, algorithm, args))
    return execute_rule(spec, nothing, rule_algorithm(spec, algorithm), ctx, args, ann, target)
end

"""
    message_passing_rule!(output, node, target, algorithm, args[, ctx, ann])

Resolve an in-place message rule and run it into `output`.
"""
@inline function message_passing_rule!(output, node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_message_rule(node, target, algorithm, args))
    spec.inplace || throw(ArgumentError("the rule for $node towards $target has no in-place form"))
    return execute_rule(spec, output, rule_algorithm(spec, algorithm), ctx, args, ann, target)
end

"""
    message_passing_marginalrule(node, cluster, algorithm, args[, ctx, ann])

Resolve the marginal rule for `cluster` and run it.
"""
@inline function message_passing_marginalrule(node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_marginal_rule(node, cluster, algorithm, args))
    return execute_rule(spec, nothing, rule_algorithm(spec, algorithm), ctx, args, ann, cluster)
end

"""
    message_passing_marginalrule!(output, node, cluster, algorithm, args[, ctx, ann])
"""
@inline function message_passing_marginalrule!(output, node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_marginal_rule(node, cluster, algorithm, args))
    spec.inplace || throw(ArgumentError("the marginal rule for $node over $cluster has no in-place form"))
    return execute_rule(spec, output, rule_algorithm(spec, algorithm), ctx, args, ann, cluster)
end

"""
    message_passing_average_energy(node, algorithm, args[, ctx, ann])

Resolve and compute a node's average energy.
"""
@inline function message_passing_average_energy(node, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_average_energy(node, algorithm, args))
    return execute_rule(spec, nothing, rule_algorithm(spec, algorithm), ctx, args, ann, nothing)
end

"""
    missing_services(spec::RuleSpec, ctx::RuleContext)

The context services a rule declares that `ctx` does not supply: names it has no entry for.
An entry whose value is `nothing`, such as an unset `matrix_correction`, is supplied. Meant to be
checked once, when a node is set up, rather than on every call.
"""
missing_services(spec::RuleSpec, ctx::RuleContext) =
    filter(service -> !haskey(getfield(ctx, :services), service), spec.services)


"""
    default_inputs_match(args::RuleArgs, ::Val{required}, ::Type{types})

Whether `args` holds each input a `default` rule names, `(container, key, selection)`, with its
type in `types`, for a single interface, a cluster or a whole group. Computed from the types
alone, so it folds to a constant.
"""
@generated function default_inputs_match(args::RuleArgs{M, Q}, ::Val{required}, ::Type{types}) where {M, Q, required, types}
    mnames, mtypes = M.parameters[1], M.parameters[2].parameters
    qnames, qtypes = Q.parameters[1], Q.parameters[2].parameters
    jkeys, jtypes = Q.parameters[3], Q.parameters[4].parameters
    function held(container, key, selection, type)
        if selection === :cluster
            position = findfirst(==(key), jkeys)
            return position !== nothing && jtypes[position] <: type
        end
        names, types = container === :m ? (mnames, mtypes) : (qnames, qtypes)
        position = findfirst(==(key), names)
        position === nothing && return false
        selection === :single && return types[position] <: type
        return types[position] <: Tuple && all(t -> t <: Union{Nothing, type}, types[position].parameters)
    end
    return all(((r, type),) -> held(r..., type), zip(required, types.parameters))
end

"""
    rule_inputs(node, container)

The inputs in a rule's `args.m` or `args.q` as `key => value` pairs: an interface by its name, a
member of a group as `(group, k)`, and a joint by its key, as `(:out, (:T, 1))`. Members a group
does not deliver are left out. `node` tells which names are groups. It is for a rule declared with
`default`, whose body walks the inputs the factorisation delivered.
"""
rule_inputs(node, container::Union{Messages, Marginals}) = rule_inputs(Val(interface_groups(node)), container)

# Generated from the container's type, so the pairs are a tuple of known types, with the keys
# as constants: a rule body walking them is type-stable.
@generated function rule_inputs(::Val{groups}, container::Union{Messages, Marginals}) where {groups}
    singles = container <: Messages ? :(container.values) : :(container.singles)
    names, types = container.parameters[1], container.parameters[2].parameters
    pairs = Any[]
    for (i, (key, type)) in enumerate(zip(names, types))
        if key in groups && type <: Tuple
            for (k, member) in enumerate(type.parameters)
                member === Nothing || push!(pairs, :($(QuoteNode((key, k))) => getfield(getfield($singles, $i), $k)))
            end
        else
            push!(pairs, :($(QuoteNode(key)) => getfield($singles, $i)))
        end
    end
    if container <: Marginals
        for (j, key) in enumerate(container.parameters[3])
            push!(pairs, :($(QuoteNode(key)) => getfield(container.joints, $j)))
        end
    end
    return Expr(:tuple, pairs...)
end
