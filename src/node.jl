export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export MeanField, FullFactorisation, Marginalisation, MomentMatching
export functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata
export FactorNodesCollection, getnodes, getnode_ids
export make_node, FactorNodeCreationOptions
export @node

using Rocket
using TupleTools
using MacroTools

import Rocket: getscheduler

import Base: show, +, push!, iterate, IteratorSize, IteratorEltype, eltype, length, size
import Base: getindex, setindex!, firstindex, lastindex

## Node traits

"""
    ValidNodeFunctionalForm

Trait specification for an object that can be used in model specification as a factor node.

See also: [`ReactiveMP.as_node_functional_form`](@ref), [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref)
"""
struct ValidNodeFunctionalForm end

"""
    UndefinedNodeFunctionalForm

Trait specification for an object that can **not** be used in model specification as a factor node.

See also: [`ReactiveMP.as_node_functional_form`](@ref), [`ReactiveMP.ValidNodeFunctionalForm`](@ref)
"""
struct UndefinedNodeFunctionalForm end

"""
    as_node_functional_form(object)

Determines `object` node functional form trait specification.
Returns either `ValidNodeFunctionalForm()` or `UndefinedNodeFunctionalForm()`.

See also: [`ReactiveMP.ValidNodeFunctionalForm`](@ref), [`ReactiveMP.UndefinedNodeFunctionalForm`](@ref)
"""
function as_node_functional_form end

as_node_functional_form(some) = UndefinedNodeFunctionalForm()

## Node types

"""
    Deterministic

`Deterministic` object used to parametrize factor node object with determinstic type of relationship between variables.

See also: [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref), [`sdtype`](@ref)
"""
struct Deterministic end

"""
    Stochastic

`Stochastic` object used to parametrize factor node object with stochastic type of relationship between variables.

See also: [`Deterministic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref), [`sdtype`](@ref)
"""
struct Stochastic end

"""
    isdeterministic(node)

Function used to check if factor node object is deterministic or not. Returns true or false.

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isstochastic`](@ref), [`sdtype`](@ref)
"""
function isdeterministic end

"""
    isstochastic(node)

Function used to check if factor node object is stochastic or not. Returns true or false.

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref), [`sdtype`](@ref)
"""
function isstochastic end

isdeterministic(::Deterministic)       = true
isdeterministic(::Type{Deterministic}) = true
isdeterministic(::Stochastic)          = false
isdeterministic(::Type{Stochastic})    = false

isstochastic(::Stochastic)          = true
isstochastic(::Type{Stochastic})    = true
isstochastic(::Deterministic)       = false
isstochastic(::Type{Deterministic}) = false

"""
    sdtype(object)

Returns either `Deterministic` or `Stochastic` for a given object (if defined).

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref)
"""
function sdtype end

# Any `Type` is considered to be a deterministic mapping unless stated otherwise (By convention, any `Distribution` type is not deterministic)
# E.g. `Matrix` is not an instance of the `Function` abstract type, however we would like to pretend it is a deterministic function
sdtype(::Type{T}) where {T}    = Deterministic()
sdtype(::Type{<:Distribution}) = Stochastic()
sdtype(::Function)             = Deterministic()

"""
    as_node_symbol(type)

Returns a symbol associated with a node `type`.
"""
function as_node_symbol end

as_node_symbol(fn::F) where {F <: Function} = Symbol(fn)

## Generic factorisation constraints

"""
    MeanField

Generic factorisation constraint used to specify a mean-field factorisation for recognition distribution `q`.

See also: [`FullFactorisation`](@ref)
"""
struct MeanField end

"""
    FullFactorisation

Generic factorisation constraint used to specify a full factorisation for recognition distribution `q`.

See also: [`MeanField`](@ref)
"""
struct FullFactorisation end

"""
    collect_factorisation(nodetype, factorisation)

This function converts given factorisation to a correct internal factorisation representation for a given node.

See also: [`MeanField`](@ref), [`FullFactorisation`](@ref)
"""
function collect_factorisation end

"""
    collect_meta(nodetype, meta)

This function converts given meta object to a correct internal meta representation for a given node.
Fallbacks to `default_meta` in case if meta is `nothing`.

See also: [`default_meta`](@ref), [`FactorNode`](@ref)
"""
function collect_meta end

collect_meta(T::Any, ::Nothing) = default_meta(T)
collect_meta(T::Any, meta::Any) = meta

"""
    default_meta(nodetype)

Returns default meta object for a given node type.

See also: [`collect_meta`](@ref), [`FactorNode`](@ref)
"""
function default_meta end

default_meta(any) = nothing

## NodeInterface

## NodeInterface constraints

"""
    AbstractInterfaceLocalConstraint

Every edge's interface may have a local "mathematical" constraint, which represent certain properties that the resulting outbound message should obey.
"""
abstract type AbstractInterfaceLocalConstraint end

struct Marginalisation <: AbstractInterfaceLocalConstraint end
struct MomentMatching <: AbstractInterfaceLocalConstraint end

is_marginalisation(::AbstractInterfaceLocalConstraint) = false
is_marginalisation(::Marginalisation)                  = true

is_moment_matching(::AbstractInterfaceLocalConstraint) = false
is_moment_matching(::MomentMatching)                   = true

interface_default_local_constraint(fform, edge) = Marginalisation()

"""
    NodeInterface

`NodeInterface` object represents a single node-variable connection.

See also: [`name`](@ref), [`tag`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
mutable struct NodeInterface
    name               :: Symbol
    local_constraint   :: AbstractInterfaceLocalConstraint
    m_out              :: MessageObservable{AbstractMessage}
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    NodeInterface(name::Symbol, local_constraint::AbstractInterfaceLocalConstraint) = new(name, local_constraint, MessageObservable(AbstractMessage), nothing, 0)
end

Base.show(io::IO, interface::NodeInterface) = print(io, string("Interface(", name(interface), ", ", local_constraint(interface), ")"))

"""
    name(interface)

Returns a name of the interface.

See also: [`NodeInterface`](@ref), [`tag`](@ref)
"""
name(symbol::Symbol) = symbol
name(interface::NodeInterface) = name(interface.name)

"""
    local_constraint(interface)

Returns a local constraint of the interface.

See also: [`AbstractInterfaceLocalConstraint`](@ref), [`Marginalisation`](@ref), [`MomentMatching`](@ref)
"""
local_constraint(interface::NodeInterface) = interface.local_constraint

"""
    tag(interface)

Returns a tag of the interface in the form of `Val{ name(interface) }`.
The major difference between tag and name is that it is possible to dispath on interface's tag in message computation rule.

See also: [`NodeInterface`](@ref), [`name`](@ref)
"""
tag(interface::NodeInterface) = Val{name(interface)}()

"""
    messageout(interface)

Returns an outbound messages stream from the given interface.

See also: [`NodeInterface`](@ref), [`messagein`](@ref)
"""
messageout(interface::NodeInterface) = interface.m_out

"""
    messagein(interface)

Returns an inbound messages stream from the given interface.

See also: [`NodeInterface`](@ref), [`messageout`](@ref)
"""
messagein(interface::NodeInterface) = messagein(interface, connectedvar(interface), connectedvarindex(interface))

messagein(interface::NodeInterface, variable::AbstractVariable, index::Int) = messageout(variable, index)
messagein(interface::NodeInterface, variable::Nothing, index::Int)          = error("`messagein` is not defined for interface $(interface). Interface has no connected variable.")

"""
    connectvariable!(interface, variable, index)

Connects a variable with the interface and given index. Index is used to distinguish this connection from others in case if variable is connected to multiple interfaces.

See also: [`NodeInterface`](@ref), [`connectedvar`](@ref), [`connectedvarindex`](@ref)
"""
function connectvariable!(interface::NodeInterface, variable, index)
    interface.connected_variable = variable
    interface.connected_index    = index
end

"""
    connectedvar(interface)

Returns connected variable for the interface.

See also: [`NodeInterface`](@ref), [`connectvariable!`](@ref), [`connectedvarindex`](@ref)
"""
connectedvar(interface::NodeInterface) = interface.connected_variable

"""
    connectedvarindex(interface)

Returns an index of connected variable for the interface.

See also: [`NodeInterface`](@ref), [`connectvariable!`](@ref), [`connectedvar`](@ref)
"""
connectedvarindex(interface::NodeInterface) = interface.connected_index

"""
    get_pipeline_stages(interface)

Returns an instance of pipeline stages of connected variable for the given interface

See also: [`NodeInterface`](@ref), [`connectvariable!`](@ref), [`connectedvar`](@ref), [`add_inbound_pipeline_stage!`](@ref)
"""
get_pipeline_stages(interface::NodeInterface) = get_pipeline_stages(connectedvar(interface))

"""
    IndexedNodeInterface

`IndexedNodeInterface` object represents a repetative node-variable connection.
Used in cases when a node may connect to a different number of random variables with the same name, e.g. means and precisions of a Gaussian Mixture node.

See also: [`name`](@ref), [`tag`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
struct IndexedNodeInterface
    index     :: Int
    interface :: NodeInterface
end

Base.show(io::IO, interface::IndexedNodeInterface) = print(io, string("IndexedInterface(", name(interface), ", ", local_constraint(interface), ", ", index(interface), ")"))

name(interface::IndexedNodeInterface)             = name(interface.interface)
local_constraint(interface::IndexedNodeInterface) = local_constraint(interface.interface)
index(interface::IndexedNodeInterface)            = interface.index
tag(interface::IndexedNodeInterface)              = (tag(interface.interface), index(interface))

messageout(interface::IndexedNodeInterface) = messageout(interface.interface)
messagein(interface::IndexedNodeInterface)  = messagein(interface.interface)

connectvariable!(interface::IndexedNodeInterface, variable, index) = connectvariable!(interface.interface, variable, index)
connectedvar(interface::IndexedNodeInterface)                      = connectedvar(interface.interface)
connectedvarindex(interface::IndexedNodeInterface)                 = connectedvarindex(interface.interface)
get_pipeline_stages(interface::IndexedNodeInterface)               = get_pipeline_stages(interface.interface)

"""
Some nodes use `IndexedInterface`, `ManyOf` structure reflects a collection of marginals from the collection of `IndexedInterface`s. `@rule` macro 
also treats `ManyOf` specially.
"""
struct ManyOf{T}
    collection::T
end

Base.show(io::IO, manyof::ManyOf) = print(io, "ManyOf(", join(manyof.collection, ",", ""), ")")

Rocket.getrecent(many::ManyOf) = ManyOf(getrecent(many.collection))

getdata(many::ManyOf)    = getdata(many.collection)
is_clamped(many::ManyOf) = is_clamped(many.collection)
is_initial(many::ManyOf) = is_initial(many.collection)
typeofdata(many::ManyOf) = typeof(ManyOf(many.collection))

paramfloattype(many::ManyOf) = paramfloattype(many.collection)

rule_method_error_type_nameof(::Type{T}) where {N, R, V <: NTuple{N, <:R}, T <: ManyOf{V}}           = string("ManyOf{", N, ", ", rule_method_error_type_nameof(dropproxytype(R)), "}")
rule_method_error_type_nameof(::Type{T}) where {N, V <: Tuple{Vararg{R, N} where R}, T <: ManyOf{V}} = string("ManyOf{", N, ", Union{", join(map(r -> rule_method_error_type_nameof(dropproxytype(r)), fieldtypes(V)), ","), "}}")

Base.iterate(many::ManyOf)        = iterate(many.collection)
Base.iterate(many::ManyOf, state) = iterate(many.collection, state)

Base.length(many::ManyOf) = length(many.collection)

struct ManyOfObservable{S} <: Subscribable{ManyOf}
    source::S
end

Rocket.getrecent(observable::ManyOfObservable) = ManyOf(Rocket.getrecent(observable.source))

@inline function Rocket.on_subscribe!(observable::ManyOfObservable, actor)
    return subscribe!(observable.source |> map(ManyOf, (d) -> ManyOf(d)), actor)
end

function combineLatestMessagesInUpdates(indexed::NTuple{N, <:IndexedNodeInterface}) where {N}
    return ManyOfObservable(combineLatestUpdates(map((in) -> messagein(in), indexed), PushNew()))
end

## FactorNodeLocalMarginals

"""
    FactorNodeLocalMarginal

This object represents local marginals for some specific factor node.
The local marginal can be joint in case of structured factorisation.
Local to factor node marginal also can be shared with a corresponding marginal of some random variable.

See also: [`FactorNodeLocalMarginals`](@ref)
"""
mutable struct FactorNodeLocalMarginal
    index  :: Int
    first  :: Int
    name   :: Symbol
    stream :: Union{Nothing, AbstractSubscribable{<:Marginal}}

    FactorNodeLocalMarginal(index::Int, first::Int, name::Symbol) = new(index, first, name, nothing)
end

# `First` defines the index of the first element in the joint marginal
# E.g. if the set of variables is (x, y, z, w) and joint is `z_w`, first is equal to 3
Base.first(localmarginal::FactorNodeLocalMarginal) = localmarginal.first

index(localmarginal::FactorNodeLocalMarginal) = localmarginal.index
name(localmarginal::FactorNodeLocalMarginal)  = localmarginal.name
tag(localmarginal::FactorNodeLocalMarginal)   = Val{name(localmarginal)}()

getstream(localmarginal::FactorNodeLocalMarginal)              = localmarginal.stream
setstream!(localmarginal::FactorNodeLocalMarginal, observable) = localmarginal.stream = observable

"""
    FactorNodeLocalMarginals

This object acts as an iterable and indexable proxy for local marginals for some node.
"""
struct FactorNodeLocalMarginals{M}
    marginals::M
end

function FactorNodeLocalMarginals(variablenames, factorisation)
    marginal_names = map(fcluster -> clustername(map(i -> variablenames[i], fcluster)), factorisation)
    index          = 0 # its better not to use zip or enumerate here to preserve tuple-like structure
    marginals      = map(marginal_names) do mname
        index += 1
        return FactorNodeLocalMarginal(index, first(factorisation[index]), mname)
    end
    return FactorNodeLocalMarginals(marginals)
end

## AbstractFactorNode

abstract type AbstractFactorNode end

isstochastic(factornode::AbstractFactorNode)    = isstochastic(sdtype(factornode))
isdeterministic(factornode::AbstractFactorNode) = isdeterministic(sdtype(factornode))

interfaceindices(factornode::AbstractFactorNode, iname::Symbol)                       = interfaceindices(factornode, (iname,))
interfaceindices(factornode::AbstractFactorNode, inames::NTuple{N, Symbol}) where {N} = map(iname -> interfaceindex(factornode, iname), inames)

## Generic Factor Node

struct FactorNodeCreationOptions{F, M, P}
    factorisation :: F
    metadata      :: M
    pipeline      :: P
end

FactorNodeCreationOptions() = FactorNodeCreationOptions(nothing, nothing, nothing)

factorisation(options::FactorNodeCreationOptions) = options.factorisation
metadata(options::FactorNodeCreationOptions)      = options.metadata
getpipeline(options::FactorNodeCreationOptions)   = options.pipeline

Base.broadcastable(options::FactorNodeCreationOptions) = Ref(options)

struct FactorNode{F, I, C, M, A, P} <: AbstractFactorNode
    fform          :: F
    interfaces     :: I
    factorisation  :: C
    localmarginals :: M
    metadata       :: A
    pipeline       :: P
end

function FactorNode(fform::Type{F}, interfaces::I, factorisation::C, localmarginals::M, metadata::A, pipeline::P) where {F, I, C, M, A, P}
    return FactorNode{Type{F}, I, C, M, A, P}(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
end

function FactorNode(fform, varnames::NTuple{N, Symbol}, factorisation, metadata, pipeline) where {N}
    interfaces     = map(varname -> NodeInterface(varname, interface_default_local_constraint(fform, varname)), varnames)
    localmarginals = FactorNodeLocalMarginals(varnames, factorisation)
    return FactorNode(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
end

function Base.show(io::IO, factornode::FactorNode)
    println(io, "FactorNode:")
    println(io, string(" form            : ", functionalform(factornode)))
    println(io, string(" sdtype          : ", sdtype(factornode)))
    println(io, string(" interfaces      : ", interfaces(factornode)))
    println(io, string(" connectedvars   : ", map(indexed_name, connectedvars(factornode))))
    println(io, string(" factorisation   : ", factorisation(factornode)))
    println(io, string(" local marginals : ", localmarginalnames(factornode)))
    println(io, string(" metadata        : ", metadata(factornode)))
    println(io, string(" pipeline        : ", getpipeline(factornode)))
end

functionalform(factornode::FactorNode)     = factornode.fform
sdtype(factornode::FactorNode)             = sdtype(functionalform(factornode))
interfaces(factornode::FactorNode)         = factornode.interfaces
connectedvars(factornode::FactorNode)      = map((i) -> connectedvar(i), interfaces(factornode))
factorisation(factornode::FactorNode)      = factornode.factorisation
localmarginals(factornode::FactorNode)     = factornode.localmarginals.marginals
localmarginalnames(factornode::FactorNode) = map(name, localmarginals(factornode))
metadata(factornode::FactorNode)           = factornode.metadata
getpipeline(factornode::FactorNode)        = factornode.pipeline

function nodefunction(factornode::FactorNode)
    return let fform = functionalform(factornode)
        (out, inputs...) -> pdf(convert(fform, inputs...), out)
    end
end

clustername(cluster) = mapreduce(v -> name(v), (a, b) -> Symbol(a, :_, b), cluster)

# Cluster is reffered to a tuple of node interfaces
clusters(factornode::FactorNode) = map(factor -> map(i -> @inbounds(interfaces(factornode)[i]), factor), factorisation(factornode))

clusterindex(factornode::FactorNode, v::Symbol)                                = clusterindex(factornode, (v,))
clusterindex(factornode::FactorNode, vindex::Int)                              = clusterindex(factornode, (vindex,))
clusterindex(factornode::FactorNode, vars::NTuple{N, NodeInterface}) where {N} = clusterindex(factornode, map(v -> name(v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Symbol}) where {N}        = clusterindex(factornode, map(v -> interfaceindex(factornode, v), vars))

# We assume that `inbound` interfaces have indices 2..N
inboundinterfaces(factornode::FactorNode)  = TupleTools.deleteat(interfaces(factornode), 1)
inboundclustername(factornode::FactorNode) = clustername(inboundinterfaces(factornode))

function interfaceindex(factornode::FactorNode, iname::Symbol)
    iindex = findfirst(interface -> name(interface) === iname, interfaces(factornode))
    return iindex !== nothing ? iindex : error("Unknown interface ':$(iname)' for $(functionalform(factornode)) node")
end

function clusterindex(factornode::FactorNode, vars::NTuple{N, Int}) where {N}
    cindex = findfirst(cluster -> all(v -> v ∈ cluster, vars), factorisation(factornode))
    return cindex !== nothing ? cindex : error("Unknown cluster '$(vars)' for $(functionalform(factornode)) node")
end

function varclusterindex(cluster, iindex::Int)
    vcindex = findfirst(index -> index === iindex, cluster)
    return vcindex !== nothing ? vcindex : error("Invalid cluster '$(vars)' for a given interface index '$(iindex)'")
end

getinterface(factornode::FactorNode, iname::Symbol) = @inbounds interfaces(factornode)[interfaceindex(factornode, iname)]

getclusterinterfaces(factornode::FactorNode, cindex::Int) = @inbounds map(i -> interfaces(factornode)[i], factorisation(factornode)[cindex])

iscontain(factornode::FactorNode, iname::Symbol) = findfirst(interface -> name(interface) === iname, interfaces(factornode)) !== nothing
isfactorised(factornode::FactorNode, factor)     = findfirst(f -> f == factor, factorisation(factornode)) !== nothing

function connect!(factornode::FactorNode, iname::Symbol, variable)
    return connect!(factornode::FactorNode, iname::Symbol, variable, getlastindex(variable))
end

function connect!(factornode::FactorNode, iname::Symbol, variable, index)
    vinterface = getinterface(factornode, iname)
    connectvariable!(vinterface, variable, index)
    setmessagein!(variable, index, messageout(vinterface))
end

## Node pipeline

abstract type AbstractNodeFunctionalDependenciesPipeline end

struct FactorNodePipeline{F <: AbstractNodeFunctionalDependenciesPipeline, S <: AbstractPipelineStage}
    functional_dependencies :: F
    extra_stages            :: S
end

get_pipeline_dependencies(pipeline::FactorNodePipeline) = pipeline.functional_dependencies
get_pipeline_stages(pipeline::FactorNodePipeline)       = pipeline.extra_stages

function Base.show(io::IO, pipeline::FactorNodePipeline)
    print(io, "FactorNodePipeline(functional_dependencies = $(pipeline.functional_dependencies), extra_stages = $(pipeline.extra_stages)")
end

function collect_pipeline end

collect_pipeline(T::Any, ::Nothing)                                       = FactorNodePipeline(default_functional_dependencies_pipeline(T), EmptyPipelineStage())
collect_pipeline(T::Any, stage::AbstractPipelineStage)                    = FactorNodePipeline(default_functional_dependencies_pipeline(T), stage)
collect_pipeline(T::Any, fdp::AbstractNodeFunctionalDependenciesPipeline) = FactorNodePipeline(fdp, EmptyPipelineStage())
collect_pipeline(T::Any, pipeline::FactorNodePipeline)                    = pipeline

## Nodes collection

struct FactorNodesCollection
    nodes    :: Vector{AbstractFactorNode}
    node_ids :: Set{Symbol}

    FactorNodesCollection() = new(Vector{AbstractFactorNode}(), Set{Symbol}())
end

function Base.show(io::IO, collection::FactorNodesCollection)
    print(io, "FactorNodesCollection(nodes: ", length(collection.nodes), ")")
end

function Base.push!(collection::FactorNodesCollection, node::AbstractFactorNode)
    push!(collection.nodes, node)
    push!(collection.node_ids, as_node_symbol(functionalform(node)))
    return node
end

function Base.push!(collection::FactorNodesCollection, nodes::AbstractArray{AbstractFactorNode})
    append!(collection.nodes, nodes)
    union!(collection.node_ids, Set(Base.Generator((node) -> as_node_symbol(functionalform(node)), nodes)))
    return nodes
end

function Base.push!(collection::FactorNodesCollection, nodes::AbstractArray{N}) where {N <: AbstractFactorNode}
    append!(collection.nodes, nodes)
    push!(collection.node_ids, as_node_symbol(functionalform(first(nodes))))
    return nodes
end

getnodes(collection::FactorNodesCollection)    = collection.nodes
getnode_ids(collection::FactorNodesCollection) = collection.node_ids

hasnodeid(collection::FactorNodesCollection, nodeid::Symbol) = nodeid ∈ getnode_ids(collection)

Base.iterate(collection::FactorNodesCollection)        = iterate(getnodes(collection))
Base.iterate(collection::FactorNodesCollection, state) = iterate(getnodes(collection), state)

Base.IteratorSize(::Type{FactorNodesCollection})   = Base.HasLength()
Base.IteratorEltype(::Type{FactorNodesCollection}) = Base.HasEltype()

Base.eltype(::Type{FactorNodesCollection}) = AbstractFactorNode

Base.length(collection::FactorNodesCollection)        = length(getnodes(collection))
Base.size(collection::FactorNodesCollection, dims...) = size(getnodes(collection), dims...)

## Functional Dependencies

function message_dependencies end
function marginal_dependencies end

Base.:+(left::AbstractNodeFunctionalDependenciesPipeline, right::AbstractPipelineStage) = FactorNodePipeline(left, right)
Base.:+(left::FactorNodePipeline, right::AbstractPipelineStage)                         = FactorNodePipeline(left.functional_dependencies, left.extra_stages + right)

### Default

"""
    DefaultFunctionalDependencies

This pipeline translates directly to enforcing a variational message passing scheme. In order to compute a message out of some edge, this pipeline requires
messages from edges within the same edge-cluster and marginals over other edge-clusters.

See also: [`ReactiveMP.RequireMessageFunctionalDependencies`](@ref), [`ReactiveMP.RequireMarginalFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct DefaultFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

function message_dependencies(::DefaultFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # First we remove current edge index from the list of dependencies
    vdependencies = TupleTools.deleteat(varcluster, varclusterindex(varcluster, iindex))
    # Second we map interface indices to the actual interfaces
    return map(inds -> map(i -> @inbounds(nodeinterfaces[i]), inds), vdependencies)
end

function marginal_dependencies(::DefaultFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return TupleTools.deleteat(nodelocalmarginals, cindex)
end

### With inbound messages

"""
    RequireMessageFunctionalDependencies(indices::Tuple, start_with::Tuple)

The same as `DefaultFunctionalDependencies`, but in order to compute a message out of some edge also requires the inbound message on the this edge.

# Arguments

- `indices`::Tuple, tuple of integers, which indicates what edges should require inbound messages
- `start_with::Tuple`, tuple of `nothing` or `<:Distribution`, which specifies the initial inbound messages for edges in `indices`

Note: `start_with` uses `setmessage!` mechanism, hence, it can be visible by other listeners on the same edge. Explicit call to `setmessage!` overwrites whatever has been passed in `start_with`.

`@model` macro accepts a simplified construction of this pipeline:

```julia
@model function some_model()
    # ...
    y ~ NormalMeanVariance(x, τ) where {
        pipeline = RequireMessage(x = vague(NormalMeanPrecision),     τ)
                                  # ^^^                               ^^^
                                  # request 'inbound' for 'x'         we may do the same for 'τ',
                                  # and initialise with `vague(...)`  but here we skip initialisation
    }
    # ...
end
```

Deprecation warning: `RequireInboundFunctionalDependencies` has been deprecated in favor of `RequireMessageFunctionalDependencies`.

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMarginalFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct RequireMessageFunctionalDependencies{I, S} <: AbstractNodeFunctionalDependenciesPipeline
    indices    :: I
    start_with :: S
end

Base.@deprecate_binding RequireInboundFunctionalDependencies RequireMessageFunctionalDependencies

function message_dependencies(dependencies::RequireMessageFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)

    # First we find dependency index in `indices`, we use it later to find `start_with` distribution
    depindex = findfirst((i) -> i === iindex, dependencies.indices)

    # If we have `depindex` in our `indices` we include it in our list of functional dependencies. It effectively forces rule to require inbound message
    if depindex !== nothing
        # `mapindex` is a lambda function here
        output     = messagein(nodeinterfaces[iindex])
        start_with = dependencies.start_with[depindex]
        # Initialise now, if message has not been initialised before and `start_with` element is not empty
        if isnothing(getrecent(output)) && !isnothing(start_with)
            setmessage!(output, start_with)
        end
        return map(inds -> map(i -> @inbounds(nodeinterfaces[i]), inds), varcluster)
    else
        return message_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    end
end

function marginal_dependencies(::RequireMessageFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return marginal_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
end

### With marginals

"""
    RequireMarginalFunctionalDependencies(indices::Tuple, start_with::Tuple)

Similar to `DefaultFunctionalDependencies`, but in order to compute a message out of some edge also requires the posterior marginal on that edge.

# Arguments

- `indices`::Tuple, tuple of integers, which indicates what edges should require their own marginals
- `start_with::Tuple`, tuple of `nothing` or `<:Distribution`, which specifies the initial marginal for edges in `indices`

Note: `start_with` uses the `setmarginal!` mechanism, hence it can be visible to other listeners on the same edge. Explicit calls to `setmarginal!` overwrites whatever has been passed in `start_with`.

`@model` macro accepts a simplified construction of this pipeline:

```julia
@model function some_model()
    # ...
    y ~ NormalMeanVariance(x, τ) where {
        pipeline = RequireMarginal(x = vague(NormalMeanPrecision),     τ)
                                   # ^^^                               ^^^
                                   # request 'marginal' for 'x'        we may do the same for 'τ',
                                   # and initialise with `vague(...)`  but here we skip initialisation
    }
    # ...
end
```

Note: The simplified construction in `@model` macro syntax is only available in `GraphPPL.jl` of version `>2.2.0`.

See also: [`ReactiveMP.DefaultFunctionalDependencies`](@ref), [`ReactiveMP.RequireMessageFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct RequireMarginalFunctionalDependencies{I, S} <: AbstractNodeFunctionalDependenciesPipeline
    indices    :: I
    start_with :: S
end

function message_dependencies(::RequireMarginalFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    return message_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
end

function marginal_dependencies(dependencies::RequireMarginalFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # First we find dependency index in `indices`, we use it later to find `start_with` distribution
    depindex = findfirst((i) -> i === iindex, dependencies.indices)

    if depindex !== nothing
        # We create an auxiliary local marginal with non-standard index here and inject it to other standard dependencies
        extra_localmarginal = FactorNodeLocalMarginal(-1, iindex, name(nodeinterfaces[iindex]))
        vmarginal           = getmarginal(connectedvar(nodeinterfaces[iindex]), IncludeAll())
        start_with          = dependencies.start_with[depindex]
        # Initialise now, if marginal has not been initialised before and `start_with` element is not empty
        if isnothing(getrecent(vmarginal)) && !isnothing(start_with)
            setmarginal!(vmarginal, start_with)
        end
        setstream!(extra_localmarginal, vmarginal)
        default = marginal_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
        # Find insertion position (probably might be implemented more efficiently)
        insertafter = sum(first(el) < iindex ? 1 : 0 for el in default; init = 0)
        return TupleTools.insertafter(default, insertafter, (extra_localmarginal,))
    else
        return marginal_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    end
end

### Everything

"""
   RequireEverythingFunctionalDependencies

This pipeline specifies that in order to compute a message of some edge update rules request everything that is available locally.
This includes all inbound messages (including on the same edge) and marginals over all local edge-clusters (this may or may not include marginals on single edges, depends on the local factorisation constraint).

See also: [`DefaultFunctionalDependencies`](@ref), [`RequireMessageFunctionalDependencies`](@ref), [`RequireMarginalFunctionalDependencies`](@ref)
"""
struct RequireEverythingFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

function ReactiveMP.message_dependencies(::RequireEverythingFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # Return all node interfaces including the edge we are trying to compute a message on
    return nodeinterfaces
end

function ReactiveMP.marginal_dependencies(::RequireEverythingFunctionalDependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    # Returns only local marginals based on local q factorisation, it does not return all possible combinations of all joint posterior marginals
    return nodelocalmarginals
end

###

default_functional_dependencies_pipeline(_) = DefaultFunctionalDependencies()

### Generic `functional_dependencies` for `AbstractFactorNode`

function functional_dependencies(factornode::AbstractFactorNode, iname::Symbol)
    return functional_dependencies(get_pipeline_dependencies(getpipeline(factornode)), factornode, iname)
end

function functional_dependencies(factornode::AbstractFactorNode, iindex::Int)
    return functional_dependencies(get_pipeline_dependencies(getpipeline(factornode)), factornode, iindex)
end

### `FactorNode` implementation of `functional_dependencies`

function functional_dependencies(dependencies, factornode::FactorNode, iname::Symbol)
    return functional_dependencies(dependencies, factornode, interfaceindex(factornode, iname))
end

function functional_dependencies(dependencies, factornode::FactorNode, iindex::Int)
    cindex = clusterindex(factornode, iindex)

    nodeinterfaces     = interfaces(factornode)
    nodeclusters       = factorisation(factornode)
    nodelocalmarginals = localmarginals(factornode)

    varcluster = @inbounds nodeclusters[cindex]

    messages  = message_dependencies(dependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
    marginals = marginal_dependencies(dependencies, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)

    return tuple(messages...), tuple(marginals...)
end

function get_messages_observable(factornode, messages)
    if !isempty(messages)
        msgs_names      = Val{map(name, messages)}()
        msgs_observable = combineLatestUpdates(map(m -> messagein(m), messages), PushNew())
        return msgs_names, msgs_observable
    else
        return nothing, of(nothing)
    end
end

function get_marginals_observable(factornode, marginals)
    if !isempty(marginals)
        marginal_names       = Val{map(name, marginals)}()
        marginals_streams    = map(marginal -> getmarginal!(factornode, marginal, IncludeAll()), marginals)
        marginals_observable = combineLatestUpdates(marginals_streams, PushNew())
        return marginal_names, marginals_observable
    else
        return nothing, of(nothing)
    end
end

# options here must implement at least `ReactiveMP.get_pipeline_stages`, `Rocket.getscheduler` and `ReactiveMP.getaddons` functions
function activate!(factornode::AbstractFactorNode, options)
    pipeline_stages            = get_pipeline_stages(options)
    scheduler                  = getscheduler(options)
    addons                     = getaddons(options)
    fform                      = functionalform(factornode)
    meta                       = metadata(factornode)
    node_pipeline              = getpipeline(factornode)
    node_pipeline_extra_stages = get_pipeline_stages(node_pipeline)

    for (iindex, interface) in enumerate(interfaces(factornode))
        cvariable = connectedvar(interface)
        if cvariable !== nothing && (israndom(cvariable) || isdata(cvariable))
            message_dependencies, marginal_dependencies = functional_dependencies(factornode, iindex)

            msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
            marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

            vtag        = tag(interface)
            vconstraint = local_constraint(interface)

            vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())  # TODO check PushEach
            vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

            mapping = let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, addons, node_if_required(fform, factornode))
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

            vmessageout = vmessageout |> map(AbstractMessage, mapping)
            vmessageout = apply_pipeline_stage(pipeline_stages, factornode, vtag, vmessageout)
            vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
            vmessageout = vmessageout |> schedule_on(scheduler)

            connect!(messageout(interface), vmessageout)
        elseif cvariable === nothing
            error("Empty variable on interface $(interface) of node $(factornode)")
        end
    end
end

function setmarginal!(factornode::FactorNode, cname::Symbol, marginal)
    lindex = findnext(lmarginal -> name(lmarginal) === cname, localmarginals(factornode), 1)
    @assert lindex !== nothing "Invalid local marginal id: $cname"
    lmarginal = @inbounds localmarginals(factornode)[lindex]
    setmarginal!(getstream(lmarginal), marginal)
end

# Here for user convenience and to be consistent with variables we don't put '!' at the of the function name
# However the underlying function may modify `factornode`, see `getmarginal!`
# In contrast with internal `getmarginal!` function this version uses `SkipInitial` strategy
getmarginal(factornode::FactorNode, cname::Symbol) = getmarginal(factornode, cname, SkipInitial())

function getmarginal(factornode::FactorNode, cname::Symbol, skip_strategy::MarginalSkipStrategy)
    lindex = findnext(lmarginal -> name(lmarginal) === cname, localmarginals(factornode), 1)
    @assert lindex !== nothing "Invalid local marginal id: $cname"
    lmarginal = @inbounds localmarginals(factornode)[lindex]
    return getmarginal!(factornode, lmarginal, skip_strategy)
end

getmarginals(factornodes::AbstractArray{<:AbstractFactorNode}, cname::Symbol)                                      = getmarginals(factornodes, cname, SkipInitial())
getmarginals(factornodes::AbstractArray{<:AbstractFactorNode}, cname::Symbol, skip_strategy::MarginalSkipStrategy) = collectLatest(map(n -> getmarginal(n, cname, skip_strategy), factornodes))

function getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal)
    return getmarginal!(factornode, localmarginal, IncludeAll())
end

function getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal, skip_strategy::MarginalSkipStrategy)
    cached_stream = getstream(localmarginal)

    if cached_stream !== nothing
        return apply_skip_filter(cached_stream, skip_strategy)
    end

    clusterindex = index(localmarginal)

    marginalname = name(localmarginal)
    marginalsize = @inbounds length(factorisation(factornode)[clusterindex])

    if marginalsize === 1
        # Cluster contains only one variable, we can take marginal over this variable
        vmarginal = getmarginal(connectedvar(getinterface(factornode, marginalname)), IncludeAll())
        setstream!(localmarginal, vmarginal)
        return apply_skip_filter(vmarginal, skip_strategy)
    else
        cmarginal = MarginalObservable()
        setstream!(localmarginal, cmarginal)

        message_dependencies  = tuple(getclusterinterfaces(factornode, clusterindex)...)
        marginal_dependencies = tuple(TupleTools.deleteat(localmarginals(factornode), clusterindex)...)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        fform = functionalform(factornode)
        vtag  = tag(localmarginal)
        meta  = metadata(factornode)

        mapping = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, node_if_required(fform, factornode))
        # TODO: discontinue operator is needed for loopy belief propagation? Check
        marginalout = combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout)

        return apply_skip_filter(cmarginal, skip_strategy)
    end
end

## make_node

"""
    make_node(node)
    make_node(node, options)

Creates a factor node of a given type and options. See the list of available factor nodes below.

See also: [`@node`](@ref)

# List of available nodes:
"""
function make_node end

function interface_get_index end
function interface_get_name end

function interface_get_index(::Type{Val{Node}}, ::Type{Val{Interface}}) where {Node, Interface}
    error("Node $Node has no interface named $Interface")
end

function interface_get_name(::Type{Val{Node}}, ::Type{Val{Interface}}) where {Node, Interface}
    error("Node $Node has no interface named $Interface")
end

make_node(fform, args::Vararg{<:AbstractVariable}) = make_node(fform, FactorNodeCreationOptions(), args...)

function make_node(fform, options::FactorNodeCreationOptions, args::Vararg{<:AbstractVariable})
    error("""
          `$(fform)` is not available as a node in the inference engine. Used in `$(name(first(args))) ~ $(fform)(...)` expression.
          Use `@node` macro to add a custom factor node corresponding to `$(fform)`. See `@node` macro for additional documentation and examples.
          """)
end

# This error message should be displayed if a node receives an incompatible number of arguments
function make_node_incompatible_number_of_arguments_error(fuppertype, fbottomtype, interfaces_list, args)
    error(
        "`$(fbottomtype)` expects $(length(interfaces_list) - 1) arguments, but $(length(args) - 1) were given. Double check the `$(indexed_name(args[1])) ~ $(fbottomtype)($(join(map(indexed_name, args[begin+1:end]), ", ")))` expression."
    )
end
# end

## macro helpers

import .MacroHelpers

function correct_interfaces end

macro node(node_fform, node_type, node_interfaces, interface_aliases)
    # Assert that the node type is either Stochastic or Deterministic, and that all interfaces are symbols
    @assert node_type ∈ [:Stochastic, :Deterministic]
    @assert length(node_interfaces.args) > 0
    for interface in node_interfaces.args
        @assert isa(interface, Symbol)
    end

    # Determine whether we should dispatch on `typeof($fform)` or `Type{$node_fform}`
    if @capture(node_fform, typeof(fform_))
        dispatch_type = quote typeof($fform) end
    else
        dispatch_type = quote Type{$node_fform} end
    end

    # Define the necessary function types
    result = quote
        ReactiveMP.as_node_functional_form(::$dispatch_type)            = ReactiveMP.ValidNodeFunctionalForm()
        ReactiveMP.sdtype(::$dispatch_type)                             = (ReactiveMP.$node_type)()
    end

    # If there are any aliases, define the alias correction function
    if @capture(interface_aliases, aliases = aliases_)
        for alias in aliases.args
            result = quote
                $result
                ReactiveMP.correct_interfaces(::$dispatch_type, nt::NamedTuple{Tuple($(alias.args))}) = NamedTuple{$(Tuple(node_interfaces.args))}(values(nt))
            end
        end
    end
    
    return esc(result)
end

macro node(node_fform, node_type, node_interfaces)
    esc(quote 
        @node($node_fform, $node_type, $node_interfaces, nothing)
    end)
end