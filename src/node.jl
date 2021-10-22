export ValidNodeFunctionalForm, UndefinedNodeFunctionalForm, as_node_functional_form
export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export MeanField, FullFactorisation, collect_factorisation
export NodeInterface, IndexedNodeInterface, name, tag, messageout, messagein
export AbstractInterfaceLocalConstraint, Marginalisation, MomentMatching
export FactorNode, functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata
export iscontain, isfactorised, getinterface
export clusters, clusterindex
export connect!, activate!
export make_node, AutoVar
export DefaultFunctionalDependencies, RequireInboundFunctionalDependencies, RequireEverythingFunctionalDependencies
export @node

using Rocket
using TupleTools

import Base: show, +
import Base: getindex, setindex!, firstindex, lastindex

## Node traits

"""
    ValidNodeFunctionalForm   

Trait specification for an object that can be used in model specification as a factor node.

See also: [`as_node_functional_form`](@ref), [`UndefinedNodeFunctionalForm`](@ref)
"""
struct ValidNodeFunctionalForm end

"""
    UndefinedNodeFunctionalForm

Trait specification for an object that can **not** be used in model specification as a factor node.

See also: [`as_node_functional_form`](@ref), [`ValidNodeFunctionalForm`](@ref)
"""
struct UndefinedNodeFunctionalForm end

"""
    as_node_functional_form(object)

Determines `object` node functional form trait specification.
Returns either `ValidNodeFunctionalForm()` or `UndefinedNodeFunctionalForm()`.

See also: [`ValidNodeFunctionalForm`](@ref), [`UndefinedNodeFunctionalForm`](@ref)
"""
function as_node_functional_form end

as_node_functional_form(::Function) = ValidNodeFunctionalForm()
as_node_functional_form(some)       = UndefinedNodeFunctionalForm()

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

isdeterministic(::Deterministic)         = true
isdeterministic(::Type{ Deterministic }) = true
isdeterministic(::Stochastic)            = false
isdeterministic(::Type{ Stochastic })    = false

isstochastic(::Stochastic)            = true
isstochastic(::Type{ Stochastic })    = true
isstochastic(::Deterministic)         = false
isstochastic(::Type{ Deterministic }) = false

"""
    sdtype(object)

Returns either `Deterministic` or `Stochastic` for a given object (if defined).

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref)
"""
function sdtype end

sdtype(::Function) = Deterministic()

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

abstract type AbstractInterfaceLocalConstraint end

struct Marginalisation <: AbstractInterfaceLocalConstraint end
struct MomentMatching  <: AbstractInterfaceLocalConstraint end # TODO: WIP

is_marginalisation(::AbstractInterfaceLocalConstraint) = false
is_marginalisation(::Marginalisation)                  = true

is_moment_matching(::AbstractInterfaceLocalConstraint) = false
is_moment_matching(::MomentMatching)                   = true

default_interface_local_constraint(factornode, edge) = Marginalisation()

"""
    NodeInterface

`NodeInterface` object represents a single node-variable connection.

See also: [`name`](@ref), [`tag`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
mutable struct NodeInterface
    name               :: Symbol
    local_constraint   :: AbstractInterfaceLocalConstraint   
    m_out              :: MessageObservable{AbstractMessage}
    m_in               :: LazyObservable{Message}
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    NodeInterface(name::Symbol, local_constraint::AbstractInterfaceLocalConstraint) = new(name, local_constraint, MessageObservable(AbstractMessage), lazy(Message), nothing, 0)
end

Base.show(io::IO, interface::NodeInterface) = print(io, string("Interface(", name(interface), ", ", local_constraint(interface), ")"))

"""
    name(interface)

Returns a name of the interface.

See also: [`NodeInterface`](@ref), [`tag`](@ref)
"""
name(symbol::Symbol)              = symbol
name(interface::NodeInterface)    = name(interface.name)

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
tag(interface::NodeInterface)     = Val{ name(interface) }

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
messagein(interface::NodeInterface)  = interface.m_in

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
connectedvar(interface::NodeInterface)      = interface.connected_variable

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
Used in cases when node may connect different number of random variables with the same name, e.g. means and precisions of Gaussian Mixture node.

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
tag(interface::IndexedNodeInterface)              = (Val{ name(interface) }(), index(interface))

messageout(interface::IndexedNodeInterface) = messageout(interface.interface)
messagein(interface::IndexedNodeInterface)  = messagein(interface.interface)

connectvariable!(interface::IndexedNodeInterface, variable, index) = connectvariable!(interface.interface, variable, index)
connectedvar(interface::IndexedNodeInterface)                      = connectedvar(interface.interface)
connectedvarindex(interface::IndexedNodeInterface)                 = connectedvarindex(interface.interface)
get_pipeline_stages(interface::IndexedNodeInterface)               = get_pipeline_stages(interface.interface)

## FactorNodeLocalMarginals

"""
    FactorNodeLocalMarginal

This object represents local marginals for some specific factor node. 
Local marginal can be joint in case of structured factorisation. 
Local to factor node marginal also can be shared with a corresponding marginal of some random variable.

See also: [`FactorNodeLocalMarginals`](@ref)
"""
mutable struct FactorNodeLocalMarginal 
    index  :: Int
    name   :: Symbol
    stream :: Union{Nothing, MarginalObservable}

    FactorNodeLocalMarginal(index::Int, name::Symbol) = new(index, name, nothing)
end

index(localmarginal::FactorNodeLocalMarginal) = localmarginal.index
name(localmarginal::FactorNodeLocalMarginal)  = localmarginal.name

getstream(localmarginal::FactorNodeLocalMarginal) = localmarginal.stream
setstream!(localmarginal::FactorNodeLocalMarginal, observable::MarginalObservable) = localmarginal.stream = observable

"""
    FactorNodeLocalMarginals

This object acts as an iterable and indexable proxy for local marginals for some node. 
"""
struct FactorNodeLocalMarginals{M}
    marginals :: M
end

function FactorNodeLocalMarginals(variablenames, factorisation)
    marginal_names = map(fcluster -> clustername(map(i -> variablenames[i], fcluster)), factorisation)
    index          = 0 # its better not to use zip here to preserve tuple-like structure
    marginals      = map((mname) -> begin index += 1; FactorNodeLocalMarginal(index, mname) end, marginal_names)
    return FactorNodeLocalMarginals(marginals)
end

## AbstractFactorNode

abstract type AbstractFactorNode end

isstochastic(factornode::AbstractFactorNode)    = isstochastic(sdtype(factornode))
isdeterministic(factornode::AbstractFactorNode) = isdeterministic(sdtype(factornode))

interfaceindices(factornode::AbstractFactorNode, iname::Symbol)                     = interfaceindices(factornode, (iname, ))
interfaceindices(factornode::AbstractFactorNode, inames::NTuple{N, Symbol}) where N = map(iname -> interfaceindex(factornode, iname), inames)

## Generic Factor Node

struct FactorNode{F, I, C, M, A, P} <: AbstractFactorNode
    fform          :: F
    interfaces     :: I
    factorisation  :: C
    localmarginals :: M
    metadata       :: A
    pipeline       :: P
end

function FactorNode(fform::Type{F}, interfaces::I, factorisation::C, localmarginals::M, metadata::A, pipeline::P) where { F, I, C, M, A, P }
    return FactorNode{Type{F}, I, C, M, A, P}(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
end

function FactorNode(fform, varnames::NTuple{N, Symbol}, factorisation, metadata, pipeline) where N
    interfaces     = map(varname -> NodeInterface(varname, default_interface_local_constraint(fform, Val(varname))), varnames)
    localmarginals = FactorNodeLocalMarginals(varnames, factorisation)
    return FactorNode(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
end

function Base.show(io::IO, factornode::FactorNode)
    println(io, "FactorNode:")
    println(io, string(" form            : ", functionalform(factornode)))
    println(io, string(" sdtype          : ", sdtype(factornode)))
    println(io, string(" interfaces      : ", interfaces(factornode)))
    println(io, string(" factorisation   : ", factorisation(factornode)))
    println(io, string(" local marginals : ", localmarginalnames(factornode)))
    println(io, string(" metadata        : ", metadata(factornode)))
    println(io, string(" pipeline        : ", getpipeline(factornode)))
end

functionalform(factornode::FactorNode)            = factornode.fform
sdtype(factornode::FactorNode)                    = sdtype(functionalform(factornode))
interfaces(factornode::FactorNode)                = factornode.interfaces
factorisation(factornode::FactorNode)             = factornode.factorisation
localmarginals(factornode::FactorNode)            = factornode.localmarginals.marginals
localmarginalnames(factornode::FactorNode)        = map(name, localmarginals(factornode))
metadata(factornode::FactorNode)                  = factornode.metadata
getpipeline(factornode::FactorNode)               = factornode.pipeline

clustername(cluster) = mapreduce(v -> name(v), (a, b) -> Symbol(a, :_, b), cluster)

# Cluster is reffered to a tuple of node interfaces 
clusters(factornode::FactorNode) = map(factor -> map(i -> begin return @inbounds interfaces(factornode)[i] end, factor), factorisation(factornode))

clusterindex(factornode::FactorNode, v::Symbol)                              = clusterindex(factornode, (v, ))
clusterindex(factornode::FactorNode, vindex::Int)                            = clusterindex(factornode, (vindex, ))
clusterindex(factornode::FactorNode, vars::NTuple{N, NodeInterface}) where N = clusterindex(factornode, map(v -> name(v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Symbol})        where N = clusterindex(factornode, map(v -> interfaceindex(factornode, v), vars))

function interfaceindex(factornode::FactorNode, iname::Symbol) 
    iindex = findfirst(interface -> name(interface) === iname, interfaces(factornode))
    return iindex !== nothing ? iindex : error("Unknown interface ':$(iname)' for $(functionalform(factornode)) node")
end

function clusterindex(factornode::FactorNode, vars::NTuple{N, Int}) where N 
    cindex = findfirst(cluster -> all(v -> v ∈ cluster, vars), factorisation(factornode))
    return cindex !== nothing ? cindex : error("Unknown cluster '$(vars)' for $(functionalform(factornode)) node")
end

function varclusterindex(cluster, iindex::Int) 
    vcindex = findfirst(index -> index === iindex, cluster)
    return vcindex !== nothing ? vcindex : error("Invalid cluster '$(vars)' for a given interface index '$(iindex)'")
end

getinterface(factornode::FactorNode, iname::Symbol)   = @inbounds interfaces(factornode)[ interfaceindex(factornode, iname) ]

getclusterinterfaces(factornode::FactorNode, cindex::Int) = @inbounds map(i -> interfaces(factornode)[i], factorisation(factornode)[ cindex ])

iscontain(factornode::FactorNode, iname::Symbol)      = findfirst(interface -> name(interface) === iname, interfaces(factornode)) !== nothing
isfactorised(factornode::FactorNode, factor)          = findfirst(f -> f == factor, factorisation(factornode)) !== nothing

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

## Functional Dependencies 

function message_dependencies end
function marginal_dependencies end

Base.:+(left::AbstractNodeFunctionalDependenciesPipeline, right::AbstractPipelineStage) = FactorNodePipeline(left, right)
Base.:+(left::FactorNodePipeline, right::AbstractPipelineStage)                         = FactorNodePipeline(left.functional_dependencies, left.extra_stages + right)

### Default 

"""
    DefaultFunctionalDependencies
"""
struct DefaultFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

function message_dependencies(::DefaultFunctionalDependencies, nodeinterfaces, varcluster, iindex)
    # First we remove current edge index from the list of dependencies
    vdependencies = TupleTools.deleteat(varcluster, varclusterindex(varcluster, iindex))
    # Second we map interface indices to the actual interfaces
    return map(inds -> map(i -> begin return @inbounds nodeinterfaces[i] end, inds), vdependencies)
end

function marginal_dependencies(::DefaultFunctionalDependencies, nodelocalmarginals, varcluster, cindex)
    return TupleTools.deleteat(nodelocalmarginals, cindex)
end

### With inbound

struct RequireInboundFunctionalDependencies{I, S} <: AbstractNodeFunctionalDependenciesPipeline
    indices    :: I
    start_with :: S
end

struct InterfacePluginStartWithMessage{M, S}
    msg        :: M
    start_with :: S
end

name(p::InterfacePluginStartWithMessage)      = name(p.msg)
messagein(p::InterfacePluginStartWithMessage) = messagein(p.start_with, p)

messagein(::Nothing, p::InterfacePluginStartWithMessage) = messagein(p.msg)
messagein(something, p::InterfacePluginStartWithMessage) = messagein(p.msg) |> start_with(Message(something, false, true))

function message_dependencies(dependencies::RequireInboundFunctionalDependencies, nodeinterfaces, varcluster, iindex) 

    # First we find dependency index in `indices`, we use it later to find `start_with` distribution
    depindex = findfirst((i) -> i === iindex, dependencies.indices)

    # If we have `depindex` in our `indices` we include it in our list of functional dependencies. It effectively forces rule to require inbound message
    if depindex !== nothing
        # `mapindex` is a lambda function here
        mapindex = let nodeinterfaces = nodeinterfaces, depindex = depindex
            (i) -> begin 
                interface = @inbounds nodeinterfaces[i]
                # InterfacePluginStartWithMessage is a proxy structure for `name` and `messagein` method for an interface
                # It returns the same name but modifies `messagein` to return an observable with `start_with` operator
                return i === iindex ? InterfacePluginStartWithMessage(interface, dependencies.start_with[depindex]) : interface
            end
        end 
        return map(inds -> map(mapindex, inds), varcluster)
    else
        return message_dependencies(DefaultFunctionalDependencies(), nodeinterfaces, varcluster, iindex)
    end
end

function marginal_dependencies(::RequireInboundFunctionalDependencies, nodelocalmarginals, varcluster, cindex)
    return marginal_dependencies(DefaultFunctionalDependencies(), nodelocalmarginals, varcluster, cindex)
end

### Everything

struct RequireEverythingFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

function ReactiveMP.message_dependencies(::RequireEverythingFunctionalDependencies, nodeinterfaces, varcluster, iindex)
    # Return all node interfaces including the edge we are trying to compuate a message on
    return nodeinterfaces
end

function ReactiveMP.marginal_dependencies(::RequireEverythingFunctionalDependencies, nodelocalmarginals, varcluster, cindex)
    # Returns only local marginals based on local q factorisation, it does not return all possible combinations of all joint posterior marginals
    return nodelocalmarginals 
end

### 

default_functional_dependencies_pipeline(_) = DefaultFunctionalDependencies()

### Generic

function functional_dependencies(dependencies, factornode::FactorNode, iname::Symbol)
    return functional_dependencies(dependencies, factornode, interfaceindex(factornode, iname))
end

function functional_dependencies(dependencies, factornode::FactorNode, iindex::Int)
    cindex  = clusterindex(factornode, iindex)

    nodeinterfaces     = interfaces(factornode)
    nodeclusters       = factorisation(factornode)
    nodelocalmarginals = localmarginals(factornode)

    varcluster = @inbounds nodeclusters[ cindex ]

    messages  = message_dependencies(dependencies, nodeinterfaces, varcluster, iindex)
    marginals = marginal_dependencies(dependencies, nodelocalmarginals, varcluster, cindex)

    return tuple(messages...), tuple(marginals...)
end

function get_messages_observable(factornode, messages)
    if !isempty(messages)
        msgs_names      = Val{ map(name, messages) }
        msgs_observable = combineLatest(map(m -> messagein(m), messages), PushNew())
        return msgs_names, msgs_observable
    else
        return nothing, of(nothing)
    end
end

function get_marginals_observable(factornode, marginals)
    if !isempty(marginals)
        marginal_names       = Val{ map(name, marginals) }
        marginals_streams    = map(marginal -> getmarginal!(factornode, marginal, IncludeAll()), marginals)
        marginals_observable = combineLatestUpdates(marginals_streams, PushNew())
        return marginal_names, marginals_observable
    else 
        return nothing, of(nothing)
    end
end


# Variational Message Passing
apply_mapping(msgs_observable, marginals_observable, mapping) = (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], mapping)

# Fallback for Belief Propagation
apply_mapping(msgs_observable, marginals_observable::SingleObservable{Nothing}, mapping) = mapping

function activate!(model, factornode::AbstractFactorNode)
    fform = functionalform(factornode)
    meta  = metadata(factornode)
    node_pipeline = getpipeline(factornode)

    node_pipeline_dependencies = get_pipeline_dependencies(node_pipeline)
    node_pipeline_extra_stages = get_pipeline_stages(node_pipeline)

    for (iindex, interface) in enumerate(interfaces(factornode))
        message_dependencies, marginal_dependencies = functional_dependencies(node_pipeline_dependencies, factornode, iindex)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        vtag        = tag(interface)
        vconstraint = local_constraint(interface)
        
        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew()) # TODO check PushEach
        vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
        mapping = apply_mapping(msgs_observable, marginals_observable, mapping)

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        # set!(messageout(interface), vmessageout |> share_recent())
        # set!(messagein(interface), messageout(connectedvar(interface), connectedvarindex(interface)))
        connect!(messageout(interface), vmessageout)
        set!(messagein(interface), messageout(connectedvar(interface), connectedvarindex(interface)))
    end
end

function setmarginal!(factornode::FactorNode, cname::Symbol, marginal)
    lindex = findnext(lmarginal -> name(lmarginal) === cname, localmarginals(factornode), 1)
    @assert lindex !== nothing "Invalid local marginal id: $cname"
    lmarginal = @inbounds localmarginals(factornode)[ lindex ]
    setmarginal!(getstream(lmarginal), marginal)
end

# Here for user convenience and to be consistent with variables we don't put '!' at the of the function name
# However the underlying function may modify `factornode`, see `getmarginal!`
# In contrast with internal `getmarginal!` function this version uses `SkipInitial` strategy
getmarginal(factornode::FactorNode, cname::Symbol) = getmarginal(factornode, cname, SkipInitial())

function getmarginal(factornode::FactorNode, cname::Symbol, skip_strategy::MarginalSkipStrategy)
    lindex = findnext(lmarginal -> name(lmarginal) === cname, localmarginals(factornode), 1)
    @assert lindex !== nothing "Invalid local marginal id: $cname"
    lmarginal = @inbounds localmarginals(factornode)[ lindex ]
    return getmarginal!(factornode, lmarginal, skip_strategy)
end

getmarginals(factornodes::AbstractArray{ <: AbstractFactorNode }, cname::Symbol)                                      = getmarginals(factornodes, cname, SkipInitial())
getmarginals(factornodes::AbstractArray{ <: AbstractFactorNode }, cname::Symbol, skip_strategy::MarginalSkipStrategy) = collectLatest(map(n -> getmarginal(n, cname, skip_strategy), factornodes))

function getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal) 
    return getmarginal!(factornode, localmarginal, IncludeAll())
end

function getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal, skip_strategy::MarginalSkipStrategy)
    cached_stream = getstream(localmarginal)

    if cached_stream !== nothing
        return as_marginal_observable(cached_stream, skip_strategy)
    end

    clusterindex = index(localmarginal)

    marginalname = name(localmarginal)
    marginalsize = @inbounds length(factorisation(factornode)[ clusterindex ])

    if marginalsize === 1 
        # Cluster contains only one variable, we can take marginal over this variable
        vmarginal = getmarginal(connectedvar(getinterface(factornode, marginalname)), IncludeAll())
        setstream!(localmarginal, vmarginal)
        return as_marginal_observable(vmarginal, skip_strategy)
    else
        cmarginal = MarginalObservable()
        setstream!(localmarginal, cmarginal)

        message_dependencies  = tuple(getclusterinterfaces(factornode, clusterindex)...)
        marginal_dependencies = tuple(TupleTools.deleteat(localmarginals(factornode), clusterindex)...)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        fform       = functionalform(factornode)
        vtag        = Val{ name(localmarginal) }
        meta        = metadata(factornode)

        mapping = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, factornode)
        # TODO: discontinue operator is needed for loopy belief propagation? Check
        marginalout = combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here

        return as_marginal_observable(cmarginal, skip_strategy)
    end
end


## TODO
function conjugate_type end

## make_node

struct AutoVar
    name :: Symbol
end

getname(autovar::AutoVar) = autovar.name

function make_node end

function interface_get_index end
function interface_get_name end

function interface_get_index(::Type{ Val{ Node } }, ::Type{ Val{ Interface } }) where { Node, Interface }
    error("Node $Node has no interface named $Interface")
end

function interface_get_name(::Type{ Val{ Node } }, ::Type{ Val{ Interface } }) where { Node, Interface }
    error("Node $Node has no interface named $Interface")
end

make_node(fform, ::AutoVar, ::Vararg{ <: AbstractVariable }; kwargs...) = error("Unknown functional form '$(fform)' used for node specification.")
make_node(fform, args::Vararg{ <: AbstractVariable }; kwargs...)        = error("Unknown functional form '$(fform)' used for node specification.")

function make_node(fform::Function, autovar::AutoVar, args::Vararg{ <: ConstVariable }; kwargs...)
    var  = constvar(getname(autovar), fform(map((d) -> getconst(d), args)...))
    return nothing, var
end

function make_node(::Type{ T }, autovar::AutoVar, args::Vararg{ <: ConstVariable }; kwargs...) where T
    var  = constvar(getname(autovar), T(map((d) -> getconst(d), args)...))
    return nothing, var
end

function make_node(fform::Function, autovar::AutoVar, args::Vararg{ <: DataVariable{ <: PointMass } }; kwargs...)
    # TODO
    message_cb = let fform = fform
        (d::Tuple) -> Message(fform(d...), false, false)
    end

    subject = combineLatest(tuple(map((a) -> messageout(a, getlastindex(a)) |> map(Any, (d) -> mean(getdata(d))), args)...), PushNew()) |> map(Message, message_cb)
    var     = datavar(getname(autovar), Any, subject = subject)
    
    return nothing, var
end

# end

## macro helpers

import .MacroHelpers

"""
    @node(fformtype, sdtype, interfaces_list)

`@node` macro creates a node for a `fformtype` type object 

# Examples
```julia

struct MyNormalDistribution 
    mean :: Float64
    var  :: Float64
end

@node MyNormalDistribution Stochastic [ out, mean, var ]
```

```julia 

@node typeof(+) Deterministic [ out, in1, in2 ]
```

See also: [`make_node`](@ref), [`Stochastic`](@ref), [`Deterministic`](@ref)
"""
macro node(fformtype, sdtype, interfaces_list)

    fbottomtype = MacroHelpers.bottom_type(fformtype)
    fuppertype  = MacroHelpers.upper_type(fformtype)

    @assert sdtype ∈ [ :Stochastic, :Deterministic ] "Invalid sdtype $(sdtype). Can be either Stochastic or Deterministic."

    @capture(interfaces_list, [ interfaces_args__ ]) ||
        error("Invalid interfaces specification.")

    interfaces = map(interfaces_args) do arg
        if @capture(arg, name_Symbol)
            return (name, [])
        elseif @capture(arg, (name_Symbol, aliases = [ aliases__ ]))
            @assert all(a -> a isa Symbol && !isequal(a, name), aliases)
            return (name, aliases)
        else
            error("Interface specification should have a 'name' or (name, aliases = [ alias1, alias2,... ]) signature.")
        end
    end 

    @assert length(interfaces) !== 0 "Node should have at least one interface."
    
    names = map(d -> first(d), interfaces)
    
    names_quoted_tuple     = Expr(:tuple, map(name -> Expr(:quote, name), names)...)
    names_indices          = Expr(:tuple, map(i -> i, 1:length(names))...)
    names_splitted_indices = Expr(:tuple, map(i -> Expr(:tuple, i), 1:length(names))...)
    
    interface_args        = map(name -> :($name::AbstractVariable), names)
    interface_connections = map(name -> :(ReactiveMP.connect!(node, $(Expr(:quote, name)), $name)), names)

    # Here we create helpers function for GraphPPL.jl interfacing
    # They are used to convert interface names from `where { q = q(x, y)q(z) }` to an equivalent tuple respresentation, e.g. `((1, 2), (3, ))`
    # The general recipe to get a proper index is to call `interface_get_index(Val{ :NodeTypeName }, interface_get_name(Val{ :NodeTypeName }, Val{ :name_expression }))`
    interface_name_getters = map(enumerate(interfaces)) do (index, interface)
        name    = first(interface)
        aliases = last(interface)

        index_name_getter  = :(ReactiveMP.interface_get_index(::Type{ Val{ $(Expr(:quote, fbottomtype)) } }, ::Type{ Val{ $(Expr(:quote, name)) } }) = $(index))
        name_symbol_getter = :(ReactiveMP.interface_get_name(::Type{ Val{ $(Expr(:quote, fbottomtype)) } }, ::Type{ Val{ $(Expr(:quote, name)) } }) = $(Expr(:quote, name)))
        name_index_getter  = :(ReactiveMP.interface_get_name(::Type{ Val{ $(Expr(:quote, fbottomtype)) } }, ::Type{ Val{ $index } }) = $(Expr(:quote, name)))

        alias_getters = map(aliases) do alias
            return :(ReactiveMP.interface_get_name(::Type{ Val{ $(Expr(:quote, fbottomtype)) } }, ::Type{ Val{ $(Expr(:quote, alias)) } }) = $(Expr(:quote, name)))
        end
    
        return quote
            $index_name_getter
            $name_symbol_getter
            $name_index_getter
            $(alias_getters...)
        end
    end

    # By default every argument passed to a factorisation option of the node is transformed by 
    # `collect_factorisation` function to have a tuple like structure.
    # The default recipe is simple: for stochastic nodes we convert `FullFactorisation` and `MeanField` objects 
    # to their tuple of indices equivalents. For deterministic nodes any factorisation is replaced by a FullFactorisation equivalent
    factorisation_collectors = if sdtype === :Stochastic 
        quote
            ReactiveMP.collect_factorisation(::$fuppertype, factorisation::Tuple)           = factorisation
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.FullFactorisation) = ($names_indices, )
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.MeanField)         = $names_splitted_indices
        end
        
    elseif sdtype === :Deterministic
        quote
            ReactiveMP.collect_factorisation(::$fuppertype, factorisation::Tuple)           = ($names_indices, )
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.FullFactorisation) = ($names_indices, )
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.MeanField)         = ($names_indices, )
        end
    else
        error("Unreachable in @node macro.") 
    end

    make_node_const_mapping = if sdtype === :Stochastic
        quote
            function ReactiveMP.make_node(fform::$fuppertype, autovar::ReactiveMP.AutoVar, args::Vararg{ <: ReactiveMP.ConstVariable{ <: ReactiveMP.PointMass } }; kwargs...)
                var  = ReactiveMP.randomvar(ReactiveMP.getname(autovar))
                node = ReactiveMP.make_node(fform, var, args...; kwargs...)
                return node, var
            end
        end
    elseif sdtype === :Deterministic
        quote
            function ReactiveMP.make_node(fform::$fuppertype, autovar::ReactiveMP.AutoVar, args::Vararg{ <: ReactiveMP.ConstVariable{ <: ReactiveMP.PointMass } }; kwargs...)
                var  = ReactiveMP.constvar(ReactiveMP.getname(autovar), fform(map((d) -> ReactiveMP.getconst(d), args)...))
                return nothing, var
            end
        end
    else
        error("Unreachable in @node macro.") 
    end
    
    res = quote

        ReactiveMP.as_node_functional_form(::$fuppertype) = ReactiveMP.ValidNodeFunctionalForm()

        ReactiveMP.sdtype(::$fuppertype) = (ReactiveMP.$sdtype)()
        
        function ReactiveMP.make_node(::$fuppertype; factorisation = ($names_indices, ), meta = nothing, pipeline = nothing)
            return ReactiveMP.FactorNode($fbottomtype, $names_quoted_tuple, ReactiveMP.collect_factorisation($fbottomtype, factorisation), ReactiveMP.collect_meta($fbottomtype, meta), ReactiveMP.collect_pipeline($fbottomtype, pipeline))
        end
        
        function ReactiveMP.make_node(::$fuppertype, $(interface_args...); factorisation = ($names_indices, ), meta = nothing, pipeline = nothing)
            node = ReactiveMP.make_node($fbottomtype, factorisation = factorisation, meta = meta, pipeline = pipeline)
            $(interface_connections...)
            return node
        end

        function ReactiveMP.make_node(fform::$fuppertype, autovar::ReactiveMP.AutoVar, args::Vararg{ <: ReactiveMP.AbstractVariable }; kwargs...)
            var  = ReactiveMP.randomvar(ReactiveMP.getname(autovar))
            node = ReactiveMP.make_node(fform, var, args...; kwargs...)
            return node, var
        end

        $(make_node_const_mapping)
        $(interface_name_getters...)

        $factorisation_collectors

    end
    
    return esc(res)
end
