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
    isdeterministic(kind::Union{Deterministic, Stochastic}) -> Bool
    isdeterministic(node) -> Bool

Whether a node is [`Deterministic`](@extref MessagePassingRulesBase.Deterministic): given its kind,
as [`sdtype`](@ref) returns it, or the kind's type, or a node, a factor node or what a node is of
(`NormalMeanVariance`, `+`), asked through its [`sdtype`](@ref).

# Examples

```jldoctest
julia> isdeterministic(Deterministic()), isdeterministic(Stochastic)
(true, false)
```
"""
function isdeterministic end

"""
    isstochastic(kind::Union{Stochastic, Deterministic}) -> Bool
    isstochastic(node) -> Bool

Whether a node is [`Stochastic`](@extref MessagePassingRulesBase.Stochastic): given its kind, as
[`sdtype`](@ref) returns it, or the kind's type, or a node, a factor node or what a node is of,
asked through its [`sdtype`](@ref).
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

# A node, a factor node or what a node is of: through its kind.
isdeterministic(node) = isdeterministic(sdtype(node))
isstochastic(node) = isstochastic(sdtype(node))

"""
    sdtype(fform) -> Union{Stochastic, Deterministic}
    sdtype(factornode::FactorNode) -> Union{Stochastic, Deterministic}

The kind of a node type, or of a factor node, as its
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) declaration says:
[`Stochastic`](@extref MessagePassingRulesBase.Stochastic)`()` for a distribution,
[`Deterministic`](@extref MessagePassingRulesBase.Deterministic)`()` for a function of the inputs.
A deterministic node's clusters are its output and the joint over its inputs, and its
contribution to the free energy has no average energy.

See also [`isdeterministic`](@ref), [`isstochastic`](@ref).
"""
sdtype(fform) = MessagePassingRulesBase.sdtype(fform)

include("interfaces.jl")
include("clusters.jl")
include("dependencies.jl")

abstract type AbstractFactorNode end

"""
    FactorNode

A factor node of the graph: its node type, its interfaces in declaration order, its local
clusters, and the function it computes, when it was given one. Create one with
[`factornode`](@ref), and wire it with [`ReactiveMP.activate!`](@ref).

A node type is anything declared with
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node): a type, such as
`NormalMeanVariance`, or a function, such as `+`. A group's members are
[`ReactiveMP.IndexedNodeInterface`](@ref)s, by their index.

See also [`functionalform`](@ref), [`getinterfaces`](@ref).
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
    factornode(fform, interfaces, factorisation = nothing; nodefn = nothing) -> FactorNode

Create a factor node of type `fform` connected to the variables in `interfaces`. Each connection
allocates the variable's stream for the node's messages; nothing is computed until the node is
activated with [`ReactiveMP.activate!`](@ref).

# Arguments

- `fform`: the node type, declared with
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node);
- `interfaces`: a collection of `(key, variable)` pairs, `key` being the interface's name,
  `:out`, or `(:m, k)` for member `k` of a group `m`; a name may be an alias. Every declared
  interface is given, a group's members as `1:n`, in any order;
- `factorisation`: the clusters of the node's local marginals, a tuple of tuples of keys,
  `((:out, :μ), (:v,))`. `nothing` means one cluster over every interface. Every interface is in
  exactly one cluster. A deterministic node ignores it: its clusters are its output alone and the
  joint over its inputs.

# Keywords

- `nodefn`: the function the node computes, which a rule reaches with
  [`getnodefn`](@extref MessagePassingRulesBase.getnodefn)`(ctx.node, Target(:out))`. Default
  `nothing`, none. A node declared with `static_inputs = :fold` needs it: the members of its
  group connected to a constant or to data are folded into it (see
  [`ReactiveMP.StaticFold`](@ref)), and the others are numbered `1:n` in their order.

The interfaces and the clusters are kept in declaration order, whatever order they are given
in, so a cluster, a dependency and an emission never depend on it.

# Returns

The [`FactorNode`](@ref).

# Throws

`ArgumentError`, naming the node, when:

- `fform` is not a declared node;
- an interface is missing, unknown, given twice, or a group's members are not `1:n`;
- the node's declaration is not met: its `matched_groups` must have as many members as each
  other, each group at least `min_group_length`, and a node declared
  `factorisation = :meanfield` accepts only clusters of one interface each;
- the factorisation names an unknown interface, puts one in two clusters, or leaves one out;
- a node that folds its static inputs has no `nodefn`, or every member of its group is static.

# Examples

```julia
using ReactiveMP, StandardMessagePassingRules

x, y = randomvar(), datavar()
node = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,)))
```
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

The node type `factornode` was created with: the type or function its
[`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) declaration names,
such as `NormalMeanVariance` or `+`.
"""
functionalform(factornode::FactorNode) = factornode.fform

"""
    getinterfaces(factornode::FactorNode) -> Vector

The interfaces of `factornode`, [`ReactiveMP.NodeInterface`](@ref)s and, for a group's members,
[`ReactiveMP.IndexedNodeInterface`](@ref)s, in declaration order, a group's members in member
order.
"""
getinterfaces(factornode::FactorNode) = factornode.interfaces
getinterface(factornode::FactorNode, index) = factornode.interfaces[index]
getlocalclusters(factornode::FactorNode) = factornode.localclusters
sdtype(factornode::FactorNode) = sdtype(functionalform(factornode))

interfaceindex(factornode::FactorNode, iname::Symbol) = findfirst(interface -> name(interface) === iname, getinterfaces(factornode))

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
    ReactiveMP.FactorNodeActivationOptions(; algorithm = nothing, postprocessor = nothing, annotations = nothing, callbacks = nothing, diagnostics = EngineDiagnostics(), context = NamedTuple(), rulefallback = nothing, logscales = false)
    ReactiveMP.FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks)

What activating a [`FactorNode`](@ref) needs. The positional form gives the first four options,
the others taking their defaults.

# Keywords

- `algorithm`: the algorithm the node's rules run under. Default `nothing`, the node's own
  ([`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm)), which is
  [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)`()` for most nodes;
- `postprocessor`: the stream postprocessor applied to the node's outbound message streams and
  its joint marginal streams (see [`ReactiveMP.AbstractStreamPostprocessor`](@ref)). Default
  `nothing`, none;
- `annotations`: the annotation processors run before and after each rule, a collection of
  [`ReactiveMP.AbstractAnnotations`](@ref). Default `nothing`, none;
- `callbacks`: the handler of the rule events, [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref)
  and [`ReactiveMP.AfterMessageRuleCallEvent`](@ref) (see [`ReactiveMP.invoke_callback`](@ref)).
  Default `nothing`, none;
- `diagnostics`: the audits of the node's rules, an [`ReactiveMP.EngineDiagnostics`](@ref).
  Default all off;
- `context`: services for the node's rules, a `NamedTuple` such as `(rng = …,)`, merged over the
  engine's (see [`ReactiveMP.node_context`](@ref)): it adds any service a rule declares and
  overrides the engine's. `nothing` is `NamedTuple()`, the default. A rule declaring a service
  nobody supplies is an error when it is resolved, naming the rule and the service;
- `rulefallback`: what gives a message where no rule matches, such as
  [`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback)`()`,
  called as `rulefallback(fform, target, args)`. Default `nothing`, which makes that a
  [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError). It never replaces a
  rule that matches, and an error inside a rule propagates;
- `logscales`: whether the node's messages carry log scales (see [`getlogscale`](@ref)): each
  rule's declared one, with its inputs' ones given to the rules that read them. Default
  `false`, none, and a rule that reads its inputs' log scales is then an error.

See [Activation options](@ref lib-activation-options) for each in use.
"""
struct FactorNodeActivationOptions{A, P, N, E, C <: NamedTuple, B}
    algorithm::A
    postprocessor::P
    annotations::N
    callbacks::E
    diagnostics::EngineDiagnostics
    context::C
    rulefallback::B
    logscales::Bool
end

FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks) =
    FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks, EngineDiagnostics(), NamedTuple(), nothing, false)

FactorNodeActivationOptions(; algorithm = nothing, postprocessor = nothing, annotations = nothing, callbacks = nothing, diagnostics = EngineDiagnostics(), context = NamedTuple(), rulefallback = nothing, logscales::Bool = false) =
    FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks, diagnostics, something(context, NamedTuple()), rulefallback, logscales)

getpostprocessor(options::FactorNodeActivationOptions) = options.postprocessor
getannotations(options::FactorNodeActivationOptions) = options.annotations
getcallbacks(options::FactorNodeActivationOptions) = options.callbacks
getdiagnostics(options::FactorNodeActivationOptions) = options.diagnostics
getcontext(options::FactorNodeActivationOptions) = options.context
getrulefallback(options::FactorNodeActivationOptions) = options.rulefallback
getlogscales(options::FactorNodeActivationOptions) = options.logscales

"""
    ReactiveMP.getalgorithm(fform, options::FactorNodeActivationOptions)

The algorithm a node of type `fform` runs under: the one in `options`, or the node's default,
[`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm)`(fform)`.
"""
getalgorithm(fform, options::FactorNodeActivationOptions) = something(options.algorithm, default_algorithm(fform))

include("static_inputs.jl")

"""
    ReactiveMP.activate!(factornode::FactorNode, options::FactorNodeActivationOptions)

Wire the message and marginal streams of a factor node, after its variables are activated.

1. **Clusters**: each cluster of the factorisation gets a
   [`ReactiveMP.FactorNodeLocalMarginal`](@ref). A cluster of one interface shares the variable's
   own marginal stream; a joint, keyed by a tuple, is computed by the node's marginal rule
   ([`ReactiveMP.MarginalMapping`](@ref)).
2. **Initial messages**: a node that declares `initial_messages` has them set on its inbound
   messages, on each interface where nothing was set before.
3. **Outbound messages**: for every interface connected to a random or a data variable, the
   inputs its rule needs are combined, and each update gives a [`DeferredMessage`](@ref) computed
   by a [`ReactiveMP.MessageMapping`](@ref), which resolves and runs the rule. An interface on a
   constant sends no message.

What a rule needs is what the node's algorithm declares
([`dependencies_spec`](@extref MessagePassingRulesBase.dependencies_spec)), or else the engine's
default scheme (see [`ReactiveMP.default_dependencies`](@ref)). Its inputs are subscribed to in
the order they are declared, which in variational message passing is the update schedule. A
group reaches a rule as one tuple in member order, with `nothing` for the members it does not
depend on.

# Throws

- `ArgumentError` when the algorithm declares a free-energy partition the factorisation is not,
  or declares no dependencies for a target, or names a cluster the factorisation does not have.

See also [`ReactiveMP.FactorNodeActivationOptions`](@ref).
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
