export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export functionalform, getinterfaces
export FactorNode, factornode

using Rocket
using TupleTools

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

"""
    ReactiveMP.AbstractFactorNode

The supertype of the engine's factor nodes, [`FactorNode`](@ref): the type a graph stores its
nodes as, as RxInfer's model does.
"""
abstract type AbstractFactorNode end

# What creating a node from the same declaration, interface keys and factorisation always gives,
# since it depends on nothing else: where each processed interface comes from among the given ones,
# the element type of their vector, the clusters and their keys, and, filled in by activation, the
# dependencies each target resolves to under each dependency declaration
# (`ReactiveMP.planned_dependencies`). A graph creates the same few shapes of node many times, so
# each shape is resolved and checked once. A node that folds its static inputs depends on its
# variables' kinds as well, and has no plan.
struct CreationPlan{T, C, K}
    spec::NodeSpec
    order::Vector{Tuple{Int, Symbol, Int}}
    clusters::C
    keys::K
    dependencies::Dict{Tuple{UInt, Int}, Any}
    CreationPlan{T}(spec::NodeSpec, order, clusters::C, keys::K) where {T, C, K} =
        new{T, C, K}(spec, order, clusters, keys, Dict{Tuple{UInt, Int}, Any}())
end

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
    # the node's `CreationPlan`, shared by every node of its shape, or `nothing`
    plan::Union{Nothing, CreationPlan}

    FactorNode(fform::Type{F}, interfaces::I, localclusters::C, nodefn::N = nothing, plan = nothing) where {F, I, C, N} =
        new{Type{F}, I, C, N}(fform, interfaces, localclusters, nodefn, plan)
    FactorNode(fform::F, interfaces::I, localclusters::C, nodefn::N = nothing, plan = nothing) where {F <: Function, I, C, N} =
        new{F, I, C, N}(fform, interfaces, localclusters, nodefn, plan)
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
    plan = cached_creation_plan(fform, spec, interfaces, factorisation)
    plan === nothing || return factornode_from_plan(fform, plan, interfaces, nodefn)
    return create_factornode(fform, spec, interfaces, factorisation, nodefn)
end

function create_factornode(fform, spec::NodeSpec, interfaces, factorisation, nodefn)
    given = resolve_interfaces(fform, interfaces)
    given, statics = fold_static_inputs(fform, spec, given)
    processed = prepare_interfaces(fform, spec, given)
    check_group_lengths(fform, spec, processed)
    clusters = collect_factorisation(fform, spec, processed, factorisation)
    check_factorisation(fform, spec, clusters)
    localclusters = FactorNodeLocalClusters(processed, clusters; lone_group_joint = isdeterministic(spec.type))
    plan = spec.static_inputs === :fold ? nothing : record_creation_plan!(fform, spec, interfaces, factorisation, processed, localclusters)
    return FactorNode(fform, processed, localclusters, node_function(fform, spec, nodefn, statics), plan)
end

creation_plan(factornode::FactorNode) = factornode.plan
creation_plan(factornode) = nothing

const CREATION_PLANS = Dict{Any, CreationPlan}()
const CREATION_PLANS_LOCK = ReentrantLock()

creation_plan_key(fform, interfaces::Union{AbstractVector, Tuple}, factorisation) =
    (fform, Tuple(first(interface) for interface in interfaces), factorisation)
creation_plan_key(fform, interfaces, factorisation) = nothing

# A plan made under another declaration of the node, before it was redefined, is not used.
function cached_creation_plan(fform, spec::NodeSpec, interfaces, factorisation)
    key = creation_plan_key(fform, interfaces, factorisation)
    key === nothing && return nothing
    plan = @lock CREATION_PLANS_LOCK get(CREATION_PLANS, key, nothing)
    return (plan === nothing || plan.spec !== spec) ? nothing : plan
end

function record_creation_plan!(fform, spec::NodeSpec, interfaces, factorisation, processed, clusters::FactorNodeLocalClusters)
    key = creation_plan_key(fform, interfaces, factorisation)
    key === nothing && return nothing
    resolved = map(interface -> given_key(fform, first(interface)), Tuple(interfaces))
    order = map(processed) do interface
        (findfirst(==(interface_key(interface)), resolved), name(interface), interface isa IndexedNodeInterface ? index(interface) : 0)
    end
    plan = CreationPlan{eltype(processed)}(spec, order, getfactorization(clusters), map(name, get_node_local_marginals(clusters)))
    @lock CREATION_PLANS_LOCK CREATION_PLANS[key] = plan
    return plan
end

function factornode_from_plan(fform, plan::CreationPlan{T}, interfaces, nodefn) where {T}
    processed = Vector{T}(undef, length(plan.order))
    for (j, (i, iname, k)) in enumerate(plan.order)
        variable = last(interfaces[i])
        processed[j] = k == 0 ? NodeInterface(iname, variable) : IndexedNodeInterface(k, NodeInterface(iname, variable))
    end
    marginals = map(FactorNodeLocalMarginal, plan.keys)
    localclusters = FactorNodeLocalClusters{typeof(marginals), typeof(plan.clusters)}(marginals, plan.clusters)
    return FactorNode(fform, processed, localclusters, nodefn, plan)
end

function node_specification(fform)
    spec = try
        nodespec(fform)
    catch err
        (err isa MethodError && err.f === nodespec) || rethrow()
        nothing
    end
    spec === nothing || return spec
    throw(
        ArgumentError(
            "`$(fform)` is not a factor node: declare it with `@define_factor_node` (from `MessagePassingRulesBase`) before creating it",
        ),
    )
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

"""
    ReactiveMP.getinterface(factornode::FactorNode, index)

The interface of `factornode` at position `index` of [`getinterfaces`](@ref).
"""
getinterface(factornode::FactorNode, index) = factornode.interfaces[index]

"""
    ReactiveMP.getlocalclusters(factornode::FactorNode) -> ReactiveMP.FactorNodeLocalClusters

The clusters of `factornode`'s factorisation and their local marginals, which
[`ReactiveMP.get_node_local_marginals`](@ref) lists.
"""
getlocalclusters(factornode::FactorNode) = factornode.localclusters

# How a node shows its edges: an interface by its name, a group's member as `name[k]`, and the
# variable it connects to by its label, or a constant by its value.
interface_display(interface::NodeInterface) = string(name(interface))
interface_display(interface::IndexedNodeInterface) = string(name(interface), "[", index(interface), "]")
variable_display(variable::ConstVariable) = variable.label === nothing ? repr(variable.constant) : string(variable.label)
variable_display(variable::AbstractVariable) = variable.label === nothing ? "unnamed" : string(variable.label)
variable_kind(variable::ConstVariable) = "constant"
variable_kind(variable::DataVariable) = "data"
variable_kind(variable::RandomVariable) = "random"
variable_kind(variable::AbstractVariable) = ""

node_display(fform::Union{Type, Function}) = string(nameof(fform))
node_display(fform) = string(fform)

Base.show(io::IO, factornode::FactorNode) =
    print(io, "FactorNode(", node_display(functionalform(factornode)), ", ", join(map(interface_display, getinterfaces(factornode)), ", "), ")")

function Base.show(io::IO, ::MIME"text/plain", factornode::FactorNode)
    fform = functionalform(factornode)
    println(io, "FactorNode ", node_display(fform), " (", isstochastic(fform) ? "stochastic" : "deterministic", ")")
    interfaces = getinterfaces(factornode)
    names = map(interface_display, interfaces)
    variables = map(i -> variable_display(getvariable(i)), interfaces)
    namewidth, varwidth = maximum(length, names; init = 0), maximum(length, variables; init = 0)
    for (label, variable, interface) in zip(names, variables, interfaces)
        println(io, "  ", rpad(label, namewidth), " ── ", rpad(variable, varwidth), "  ", variable_kind(getvariable(interface)))
    end
    clusters = map(cluster -> "(" * join(map(i -> names[i], cluster), ", ") * ")", getfactorization(getlocalclusters(factornode)))
    print(io, "  clusters: ", join(clusters, " "))
    return nothing
end
sdtype(factornode::FactorNode) = sdtype(functionalform(factornode))

interfaceindex(factornode::FactorNode, iname::Symbol) = findfirst(interface -> name(interface) === iname, getinterfaces(factornode))

# The key an interface is known by in a factorisation and in an error: `:out`, or `(:m, k)`.
interface_key(interface::NodeInterface) = name(interface)
interface_key(interface::IndexedNodeInterface) = (name(interface), index(interface))

# `alias_interface` scans the node's declaration; a graph asks for the same few names of the
# same few node types many times, so the answers are cached, each with the declaration it was
# resolved under: one made before the node was redefined is not used.
const ALIAS_CACHE = Dict{Tuple{Any, Symbol}, Tuple{NodeSpec, Symbol}}()
const ALIAS_LOCK = ReentrantLock()

function cached_alias_interface(fform, name::Symbol)
    key = (fform, name)
    spec = nodespec(fform)
    cached = @lock ALIAS_LOCK get(ALIAS_CACHE, key, nothing)
    (cached === nothing || first(cached) !== spec) || return last(cached)
    resolved = MessagePassingRulesBase.alias_interface(fform, name)
    @lock ALIAS_LOCK ALIAS_CACHE[key] = (spec, resolved)
    return resolved
end

given_key(fform, name::Symbol) = cached_alias_interface(fform, name)
given_key(fform, (name, k)::Tuple{Symbol, Integer}) = (cached_alias_interface(fform, name), Int(k))
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
    return interface_vector(processed)
end

# The interfaces in a vector of one concrete type, or, for a node with groups, of the union of the
# two kinds: a small union, which the compiler splits at each call on an element, where the
# `Vector{Any}` that concatenating them gives would dispatch at run time.
function interface_vector(processed)
    all(interface -> interface isa NodeInterface, processed) && return convert(Vector{NodeInterface}, processed)
    all(interface -> interface isa IndexedNodeInterface, processed) && return convert(Vector{IndexedNodeInterface}, processed)
    return convert(Vector{Union{NodeInterface, IndexedNodeInterface}}, processed)
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
    ReactiveMP.FactorNodeActivationOptions(; algorithm = nothing, postprocessor = nothing, annotations = nothing, callbacks = nothing, diagnostics = EngineDiagnostics(), context = NamedTuple(), rulefallback = nothing, logscales = false, initial_messages = nothing)
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
  `false`, none, and a rule that reads its inputs' log scales is then an error;
- `initial_messages`: messages to start the node's inbound messages with, a `NamedTuple` keyed by
  interface name or alias, `(in = NormalMeanPrecision(0.0, 0.01),)`; a group takes a tuple, one
  message per member, `nothing` for a member left alone. Each is set on this node's own edge
  only, before inference, and takes the place of any message set there before and of one the
  node declares ([`initial_messages`](@extref MessagePassingRulesBase.initial_messages)); an
  interface on a constant is left alone. Default `nothing`, none.

See [Activation options](@ref lib-activation-options) for each in use.
"""
struct FactorNodeActivationOptions{A, P, N, E, C <: NamedTuple, B, I}
    algorithm::A
    postprocessor::P
    annotations::N
    callbacks::E
    diagnostics::EngineDiagnostics
    context::C
    rulefallback::B
    logscales::Bool
    initial_messages::I
end

FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks) =
    FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks, EngineDiagnostics(), NamedTuple(), nothing, false, nothing)

FactorNodeActivationOptions(; algorithm = nothing, postprocessor = nothing, annotations = nothing, callbacks = nothing, diagnostics = EngineDiagnostics(), context = NamedTuple(), rulefallback = nothing, logscales::Bool = false, initial_messages = nothing) =
    FactorNodeActivationOptions(algorithm, postprocessor, annotations, callbacks, diagnostics, something(context, NamedTuple()), rulefallback, logscales, initial_messages)

"""
    ReactiveMP.getpostprocessor(options::FactorNodeActivationOptions)

The `postprocessor` option: the stream postprocessor of the node's streams, or `nothing`.
"""
getpostprocessor(options::FactorNodeActivationOptions) = options.postprocessor
getannotations(options::FactorNodeActivationOptions) = options.annotations

"""
    ReactiveMP.getcallbacks(options::FactorNodeActivationOptions)

The `callbacks` option: the handler of the node's rule-call events, or `nothing`.
"""
getcallbacks(options::FactorNodeActivationOptions) = options.callbacks
getdiagnostics(options::FactorNodeActivationOptions) = options.diagnostics
getcontext(options::FactorNodeActivationOptions) = options.context
getrulefallback(options::FactorNodeActivationOptions) = options.rulefallback
getlogscales(options::FactorNodeActivationOptions) = options.logscales
getinitialmessages(options::FactorNodeActivationOptions) = options.initial_messages

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
2. **Initial messages**: the option `initial_messages` sets its messages on the node's inbound
   messages, in place of anything set before; then a node that declares `initial_messages` has
   them set on each interface where nothing is set yet.
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
    ctx = node_context(factornode, getcontext(options))
    initialize_clusters!(getlocalclusters(factornode), factornode, options, ctx)
    seed_initial_messages!(factornode, getinitialmessages(options))
    return activate_messages!(factornode, options, ctx)
end

# The messages given for this node, set on its inbound messages whatever was set before; then
# the ones the node declares, set where nothing is set yet, so a user's initialisation wins. An
# interface on a constant has none.
function seed_initial_messages!(factornode::FactorNode, given = nothing)
    given === nothing || seed_given_messages!(factornode, given)
    for (key, message) in MessagePassingRulesBase.initial_messages(functionalform(factornode))
        interface = getinterface(factornode, interfaceindex(factornode, key))
        israndom(interface) || isdata(interface) || continue
        stream = get_stream_of_inbound_messages(interface)
        Rocket.getrecent(stream) === nothing && set_initial_message!(stream, message)
    end
    return nothing
end

function seed_given_messages!(factornode::FactorNode, given::NamedTuple)
    fform, interfaces = functionalform(factornode), getinterfaces(factornode)
    for (key, message) in pairs(given)
        resolved = given_key(fform, key)
        positions = findall(interface -> name(interface) === resolved, interfaces)
        isempty(positions) && throw(ArgumentError("`$(fform)`: an initial message for `$(key)`, which this node does not connect"))
        if interfaces[first(positions)] isa IndexedNodeInterface
            message isa Union{Tuple, AbstractVector} && length(message) == length(positions) || throw(
                ArgumentError("`$(fform)`: the initial messages of the group `$(key)` are a tuple, one per member, $(length(positions)) here; got `$(repr(message))`"),
            )
            foreach((position, member) -> member === nothing || seed_message!(interfaces[position], member), positions, message)
        else
            seed_message!(interfaces[only(positions)], message)
        end
    end
    return nothing
end
seed_given_messages!(factornode::FactorNode, given) =
    throw(ArgumentError("`$(functionalform(factornode))`: `initial_messages` is a NamedTuple keyed by interface, `(in = …,)`; got `$(repr(given))`"))

function seed_message!(interface, message)
    (israndom(interface) || isdata(interface)) && set_initial_message!(get_stream_of_inbound_messages(interface), message)
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
