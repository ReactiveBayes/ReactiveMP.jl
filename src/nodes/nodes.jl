export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export functionalform, getinterfaces
export FactorNode, factornode

using Rocket
using TupleTools

import Base:
    show, +, push!, iterate, IteratorSize, IteratorEltype, eltype, length, size
import Base: getindex, setindex!, firstindex, lastindex

import MessagePassingRulesBase
import MessagePassingRulesBase: Stochastic, Deterministic, NodeSpec, nodespec, default_algorithm

"""
    isdeterministic(node)

Whether a node, its node type or its [`sdtype`](@ref) is deterministic.
"""
function isdeterministic end

"""
    isstochastic(node)

Whether a node, its node type or its [`sdtype`](@ref) is stochastic.
"""
function isstochastic end

isdeterministic(::Deterministic) = true
isdeterministic(::Type{Deterministic}) = true
isdeterministic(::Stochastic) = false
isdeterministic(::Type{Stochastic}) = false

isstochastic(::Stochastic) = true
isstochastic(::Type{Stochastic}) = true
isstochastic(::Deterministic) = false
isstochastic(::Type{Deterministic}) = false

"""
    sdtype(node)

`Stochastic()` or `Deterministic()`, as the node's declaration says.
"""
sdtype(fform) = MessagePassingRulesBase.sdtype(fform)

include("interfaces.jl")
include("clusters.jl")
include("dependencies.jl")

abstract type AbstractFactorNode end

"""
    FactorNode

A factor node in the graph: its node type `fform`, its interfaces in declaration order (a
group's members as [`ReactiveMP.IndexedNodeInterface`](@ref)s, by their index), its local
clusters, and the function it computes when it has one (see [`ReactiveMP.StaticFold`](@ref)).
A node type is anything declared with `MessagePassingRulesBase.@define_factor_node`.
"""
struct FactorNode{F, I, C, N} <: AbstractFactorNode
    fform::F
    interfaces::I
    localclusters::C
    nodefn::N

    FactorNode(fform::Type{F}, interfaces::I, localclusters::C, nodefn::N = nothing) where {F, I, C, N} =
        new{Type{F}, I, C, N}(fform, interfaces, localclusters, nodefn)
    FactorNode(fform::F, interfaces::I, localclusters::C, nodefn::N = nothing) where {F <: Function, I, C, N} =
        new{F, I, C, N}(fform, interfaces, localclusters, nodefn)
end

"""
    factornode(fform, interfaces, factorisation = nothing; nodefn = nothing)

Create a factor node of type `fform` connected to variables.

- `interfaces` pairs each interface with its variable, as `(name, variable)` for a single
  interface and `((name, k), variable)` for member `k` of a group. The names may be aliases.
  Every declared interface must be given, and a group's members as `1:n`.
- `factorisation` lists the clusters as tuples of interface keys, `((:out, :μ), (:v,))`, a
  group member being `(:m, k)`. `nothing` means one cluster over every interface. A
  deterministic node ignores it: its clusters are `out` alone and the joint over its inputs.
- `nodefn` is the function the node computes, which a rule reaches with
  `MessagePassingRulesBase.getnodefn(ctx.node, Target(:out))`.

What the node's declaration requires of the graph is checked here: its `matched_groups` must
have as many members as each other, every group at least `min_group_length`, and a node
declared `factorisation = :meanfield` accepts only clusters of one interface each.

A node declared with `static_inputs = :fold` must be given `nodefn`. The members of its group
connected to a constant or to data are folded into it: they get no interface, the others are
numbered `1:n` in their order, and every update waits for the folded values.

The interfaces are kept in declaration order, so a cluster, a dependency and an emission
never depend on the order the caller listed them in.
"""
function factornode(fform::F, interfaces, factorisation = nothing; nodefn = nothing) where {F}
    spec = node_specification(fform)
    given = resolve_interfaces(fform, interfaces)
    given, statics = fold_static_inputs(fform, spec, given)
    processed = prepare_interfaces(fform, spec, given)
    check_group_lengths(fform, spec, processed)
    clusters = collect_factorisation(fform, spec, processed, factorisation)
    check_factorisation(fform, spec, clusters)
    return FactorNode(fform, processed, FactorNodeLocalClusters(processed, clusters), node_function(fform, spec, nodefn, statics))
end

function node_specification(fform)
    applicable(nodespec, fform) || throw(
        ArgumentError(
            "`$(fform)` is not a factor node: declare it with `@define_factor_node` (from `MessagePassingRulesBase`) before creating it",
        ),
    )
    return nodespec(fform)
end

"""
    functionalform(factornode::FactorNode)

The functional form a factor node was created with: the node type or function its
`@define_factor_node` declaration names, such as `NormalMeanVariance` or `+`.
"""
functionalform(factornode::FactorNode) = factornode.fform

"""
    getinterfaces(factornode::FactorNode)

A factor node's interfaces, in declaration order, a group's members in member order.
"""
getinterfaces(factornode::FactorNode) = factornode.interfaces
getinterface(factornode::FactorNode, index) = factornode.interfaces[index]
# `getinboundinterfaces` skips the first interface, which is the output by convention
getinboundinterfaces(factornode::FactorNode) = view(
    factornode.interfaces,
    (firstindex(factornode.interfaces) + 1):lastindex(factornode.interfaces),
)
getlocalclusters(factornode::FactorNode) = factornode.localclusters
sdtype(factornode::FactorNode) = sdtype(functionalform(factornode))

interfaceindex(factornode::FactorNode, iname::Symbol) = findfirst(interface -> name(interface) === iname, getinterfaces(factornode))
interfaceindices(factornode::FactorNode, iname::Symbol) = (interfaceindex(factornode, iname),)
interfaceindices(factornode::FactorNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

# The key an interface is known by in a factorisation and in an error: `:out`, or `(:m, k)`.
interface_key(interface::NodeInterface) = name(interface)
interface_key(interface::IndexedNodeInterface) = (name(interface), index(interface))

given_key(fform, name::Symbol) = MessagePassingRulesBase.alias_interface(fform, name)
given_key(fform, (name, k)::Tuple{Symbol, Integer}) = (MessagePassingRulesBase.alias_interface(fform, name), Int(k))
given_key(fform, key) = throw(ArgumentError("an interface of `$(fform)` is `:name` or `(:group, k)`, got `$(repr(key))`"))

# The given interfaces by their resolved keys, `:out` or `(:m, k)`.
function resolve_interfaces(fform, interfaces)
    isempty(interfaces) && throw(ArgumentError("a factor node needs at least one interface; got none for `$(fform)`"))
    given = Dict{Any, Any}()
    for (key, variable) in interfaces
        resolved = given_key(fform, key)
        haskey(given, resolved) && throw(
            ArgumentError(
                "`$(fform)` has a duplicate entry for interface `$(repr(resolved))`. Did you pass an array (e.g. `x`) instead of an array element (e.g. `x[i]`)? Check your variable indices.",
            ),
        )
        given[resolved] = variable
    end
    return given
end

# Under `static_inputs = :fold`, the group members connected to a constant or to data are taken
# out, as `(position, variable)`, and the rest renumbered `1:n` in order.
fold_static_inputs(fform, spec::NodeSpec, given) =
    spec.static_inputs === :fold ? fold_static_inputs(fform, spec, given, only_group(fform, spec)) : (given, ())

function only_group(fform, spec::NodeSpec)
    groups = [i.name for i in spec.interfaces if i.group]
    length(groups) == 1 || throw(ArgumentError("`$(fform)` folds its static inputs, which needs exactly one group of inputs; it has $(Tuple(groups))"))
    return only(groups)
end

function fold_static_inputs(fform, spec::NodeSpec, given, group::Symbol)
    members = sort!([k for (g, k) in (key for key in keys(given) if key isa Tuple) if g === group])
    statics = Tuple((k, given[(group, k)]) for k in members if isconst(given[(group, k)]) || isdata(given[(group, k)]))
    free = [k for k in members if !(isconst(given[(group, k)]) || isdata(given[(group, k)]))]
    isempty(free) && !isempty(members) && throw(
        ArgumentError("`$(fform)`: every member of the group `$(group)` is a constant or data, so there is no input left to infer"),
    )
    folded = Dict{Any, Any}(key => variable for (key, variable) in given if !(key isa Tuple && first(key) === group))
    for (i, k) in enumerate(free)
        folded[(group, i)] = given[(group, k)]
    end
    return folded, statics
end

function node_function(fform, spec::NodeSpec, nodefn, statics)
    spec.static_inputs === :fold || return nodefn
    nodefn === nothing && throw(ArgumentError("`$(fform)` folds its static inputs into its function, so it needs `nodefn`, the function it computes"))
    return StaticFold(nodefn, statics)
end

function prepare_interfaces(fform, spec::NodeSpec, given::AbstractDict)
    processed = Any[]
    for declared in spec.interfaces
        if declared.group
            members = sort!([k for (key, k) in (key for key in keys(given) if key isa Tuple) if key === declared.name])
            # A group may be empty only where the node allows it, `min_group_length = 0`.
            isempty(members) && spec.min_group_length > 0 &&
                throw(ArgumentError("`$(fform)`: the group `$(declared.name)` needs at least one member, as `(:$(declared.name), 1)`"))
            members == 1:length(members) || throw(ArgumentError("`$(fform)`: the members of the group `$(declared.name)` must be `1:n`, got $(members)"))
            for k in members
                push!(processed, IndexedNodeInterface(k, NodeInterface(declared.name, pop!(given, (declared.name, k)))))
            end
        else
            haskey(given, declared.name) || throw(ArgumentError("`$(fform)` needs a variable for its interface `$(declared.name)`"))
            push!(processed, NodeInterface(declared.name, pop!(given, declared.name)))
        end
    end
    isempty(given) || throw(ArgumentError("`$(fform)` has no interfaces $(join(repr.(collect(keys(given))), ", ")); its interfaces are $(MessagePassingRulesBase.interfaces(fform))"))
    return [processed...]
end

# What the declaration requires of the groups: at least `min_group_length` members each, and
# as many members as each other within a set of `matched_groups`.
function check_group_lengths(fform, spec::NodeSpec, interfaces)
    lengths = Dict(i.name => count(interface -> interface isa IndexedNodeInterface && name(interface) === i.name, interfaces) for i in spec.interfaces if i.group)
    for (group, n) in lengths
        n >= spec.min_group_length || throw(ArgumentError("`$(fform)`: the group `$(group)` needs at least $(spec.min_group_length) members, got $(n)"))
    end
    for matched in spec.matched_groups
        counts = map(group -> lengths[group], matched)
        allequal(counts) || throw(
            ArgumentError("`$(fform)`: the groups $(join(map(g -> "`$g`", matched), " and ")) must have as many members as each other, got $(join(counts, " and "))"),
        )
    end
    return nothing
end

# A node declared `factorisation = :meanfield` accepts only clusters of one interface each.
function check_factorisation(fform, spec::NodeSpec, clusters)
    spec.factorisation === :meanfield && any(cluster -> length(cluster) > 1, clusters) && throw(
        ArgumentError("`$(fform)` accepts only a mean-field factorisation, each interface in a cluster of its own; got clusters of $(join(map(length, clusters), ", ")) interfaces"),
    )
    return nothing
end

# The clusters as tuples of positions into the processed interfaces, sorted within each
# cluster and by their first member. A deterministic node's clusters are its output alone and
# the joint over its inputs, whatever the caller's factorisation.
function collect_factorisation(fform, spec::NodeSpec, interfaces, factorisation)
    everything = (Tuple(eachindex(interfaces)),)
    isdeterministic(spec.type) && return length(interfaces) == 1 ? everything : ((1,), Tuple(2:length(interfaces)))
    factorisation === nothing && return everything
    keys = map(interface_key, interfaces)
    covered = Int[]
    clusters = map(Tuple(factorisation)) do cluster
        positions = map(Tuple(cluster)) do key
            resolved = given_key(fform, key)
            position = findfirst(==(resolved), keys)
            position === nothing && throw(ArgumentError("`$(fform)`: the factorisation names `$(repr(key))`, which is not one of its interfaces $(Tuple(keys))"))
            position in covered && throw(ArgumentError("`$(fform)`: the interface `$(repr(key))` is in more than one cluster"))
            push!(covered, position)
            position
        end
        Tuple(sort!(collect(positions)))
    end
    length(covered) == length(interfaces) || throw(
        ArgumentError("`$(fform)`: the factorisation does not cover $(join(repr.(keys[setdiff(eachindex(keys), covered)]), ", "))"),
    )
    return Tuple(sort!(collect(clusters); by = first))
end

"""
    ReactiveMP.FactorNodeActivationOptions(; algorithm, postprocessor, annotations, callbacks, diagnostics, context, rulefallback)

Everything needed to activate a [`FactorNode`](@ref):

- `algorithm` — the algorithm the node's rules run under; `nothing` means its default
  (`MessagePassingRulesBase.default_algorithm`), `DefaultAlgorithm()` for almost every node;
- `postprocessor` — an optional stream postprocessor applied to every stream created (see
  [`ReactiveMP.AbstractStreamPostprocessor`](@ref));
- `annotations` — optional annotation processors (see [`ReactiveMP.AbstractAnnotations`](@ref));
- `callbacks` — optional callbacks invoked around every rule call (see
  [`ReactiveMP.invoke_callback`](@ref));
- `diagnostics` — the audits the node's rules run under, all off by default (see
  [`ReactiveMP.EngineDiagnostics`](@ref));
- `context` — services for the node's rules, a `NamedTuple` such as `(rng = …, matrix_correction =
  …)`, merged over the engine's (see [`ReactiveMP.node_context`](@ref)): any name a rule declares
  is allowed, and a name the engine supplies is overridden;
- `rulefallback` — the message where no rule matches, such as
  `MessagePassingRulesBase.NodeFunctionRuleFallback()`; `nothing`, the default, makes that a
  `RuleNotFoundError`. It is consulted only when no rule is found, so it never replaces a rule.

Every option is a keyword with these defaults; `FactorNodeActivationOptions(algorithm,
postprocessor, annotations, callbacks)` gives the first four positionally.
"""
struct FactorNodeActivationOptions{A, P, N, E, C <: NamedTuple, B}
    algorithm::A
    postprocessor::P
    annotations::N
    callbacks::E
    diagnostics::EngineDiagnostics
    context::C
    rulefallback::B
end

FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks) =
    FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks, EngineDiagnostics(), NamedTuple(), nothing)

FactorNodeActivationOptions(; algorithm = nothing, postprocessor = nothing, annotations = nothing, callbacks = nothing, diagnostics = EngineDiagnostics(), context = NamedTuple(), rulefallback = nothing) =
    FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks, diagnostics, something(context, NamedTuple()), rulefallback)

getpostprocessor(options::FactorNodeActivationOptions) = options.postprocessor
getannotations(options::FactorNodeActivationOptions) = options.annotations
getcallbacks(options::FactorNodeActivationOptions) = options.callbacks
getdiagnostics(options::FactorNodeActivationOptions) = options.diagnostics
getcontext(options::FactorNodeActivationOptions) = options.context
getrulefallback(options::FactorNodeActivationOptions) = options.rulefallback

"""
    ReactiveMP.getalgorithm(fform, options::FactorNodeActivationOptions)

The algorithm a node of type `fform` runs under: the one in `options`, or the node's default.
"""
getalgorithm(fform, options::FactorNodeActivationOptions) = something(options.algorithm, default_algorithm(fform))

include("static_inputs.jl")

"""
    ReactiveMP.activate!(factornode::FactorNode, options::FactorNodeActivationOptions)

Wires the node's message and marginal streams into the graph.

1. **Clusters** — each cluster of the factorisation gets a
   [`ReactiveMP.FactorNodeLocalMarginal`](@ref). A cluster keyed by a single name shares the
   variable's own marginal stream; a joint, keyed by a tuple (a whole group's included, even
   of one member), is computed with the node's marginal rule.
2. **Outbound messages** — for every interface connected to a random or data variable, the
   inbound messages and local marginals the rule needs are combined with `combineLatest`, and
   each update is turned into a [`ReactiveMP.DeferredMessage`](@ref) by a
   [`ReactiveMP.MessageMapping`](@ref), which resolves and runs the rule.

What a rule needs is what the node's algorithm declares
(`MessagePassingRulesBase.dependencies_spec`), or else the engine's default scheme, driven by
the factorisation: the messages inside its own cluster, and the marginals of the other
clusters. A group reaches a rule as one tuple in member order, with `nothing` for the members
it does not depend on. Interfaces connected to constants are skipped: their message is fixed.

An algorithm that declares a free-energy partition requires the factorisation to be that
partition. A joint cluster may hold a whole group, `(:in,)`, or some of its members, keyed with
them, `(:out, (:in, 1))`.

A node that declares `initial_messages` has them set on its inbound messages here, on each
interface where nothing was set before.
"""
function activate!(factornode::FactorNode, options::FactorNodeActivationOptions)
    fform = functionalform(factornode)
    algorithm = getalgorithm(fform, options)
    spec = MessagePassingRulesBase.dependencies_spec(fform, algorithm)
    spec === nothing || check_partition(factornode, algorithm, MessagePassingRulesBase.free_energy_partition(spec))
    initialize_clusters!(getlocalclusters(factornode), factornode, options)
    seed_initial_messages!(factornode)
    return activate_messages!(factornode, options)
end

# The messages the node declares for its interfaces, set on the inbound message of each where
# nothing is set yet, so a user's initialisation wins; an interface on a constant has none.
function seed_initial_messages!(factornode::FactorNode)
    for (key, message) in MessagePassingRulesBase.initial_messages(functionalform(factornode))
        interface = getinterface(factornode, interfaceindex(factornode, key))
        israndom(interface) || isdata(interface) || continue
        stream = get_stream_of_inbound_messages(interface)
        Rocket.getrecent(stream) === nothing && set_initial_message!(stream, message)
    end
    return nothing
end

# The factorisation must be the partition the algorithm declares, block for block. A group's
# name in a block stands for all of its members.
check_partition(factornode, algorithm, ::Nothing) = nothing
function check_partition(factornode, algorithm, partition)
    interfaces = getinterfaces(factornode)
    block_positions(block) = Set(i for i in eachindex(interfaces) if name(interfaces[i]) in block)
    declared = Set(map(block_positions, partition))
    factorisation = Set(map(Set, getfactorization(getlocalclusters(factornode))))
    declared == factorisation && return nothing
    member_keys(positions) = Tuple(interface_key(interfaces[i]) for i in sort!(collect(positions)))
    throw(
        ArgumentError(
            "`$(functionalform(factornode))` runs under $(algorithm), which declares the free-energy partition $(Tuple(partition)); its factorisation $(Tuple(map(member_keys, getfactorization(getlocalclusters(factornode))))) differs",
        ),
    )
end
