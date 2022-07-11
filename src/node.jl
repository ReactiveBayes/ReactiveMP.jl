export ValidNodeFunctionalForm, UndefinedNodeFunctionalForm, as_node_functional_form
export Deterministic, Stochastic, isdeterministic, isstochastic, sdtype
export MeanField, FullFactorisation, collect_factorisation
export NodeInterface, IndexedNodeInterface, name, tag, messageout, messagein
export AbstractInterfaceLocalConstraint, Marginalisation, MomentMatching
export FactorNode,
    FactorNodeCreationOptions, functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata
export iscontain, isfactorised, getinterface
export clusters, clusterindex
export connect!, activate!
export make_node
export DefaultFunctionalDependencies,
    RequireMessageFunctionalDependencies,
    RequireMarginalFunctionalDependencies, RequireEverythingFunctionalDependencies
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

abstract type AbstractInterfaceLocalConstraint end

struct Marginalisation <: AbstractInterfaceLocalConstraint end
struct MomentMatching <: AbstractInterfaceLocalConstraint end # TODO: WIP

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
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    NodeInterface(name::Symbol, local_constraint::AbstractInterfaceLocalConstraint) =
        new(name, local_constraint, MessageObservable(AbstractMessage), nothing, 0)
end

Base.show(io::IO, interface::NodeInterface) =
    print(io, string("Interface(", name(interface), ", ", local_constraint(interface), ")"))

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
tag(interface::NodeInterface) = Val{name(interface)}

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

Base.show(io::IO, interface::IndexedNodeInterface) = print(
    io,
    string("IndexedInterface(", name(interface), ", ", local_constraint(interface), ", ", index(interface), ")")
)

name(interface::IndexedNodeInterface)             = name(interface.interface)
local_constraint(interface::IndexedNodeInterface) = local_constraint(interface.interface)
index(interface::IndexedNodeInterface)            = interface.index
tag(interface::IndexedNodeInterface)              = (Val{name(interface)}(), index(interface))

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

function FactorNode(
    fform::Type{F},
    interfaces::I,
    factorisation::C,
    localmarginals::M,
    metadata::A,
    pipeline::P
) where {F, I, C, M, A, P}
    return FactorNode{Type{F}, I, C, M, A, P}(fform, interfaces, factorisation, localmarginals, metadata, pipeline)
end

function FactorNode(fform, varnames::NTuple{N, Symbol}, factorisation, metadata, pipeline) where {N}
    interfaces     = map(varname -> NodeInterface(varname, default_interface_local_constraint(fform, Val(varname))), varnames)
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

clustername(cluster) = mapreduce(v -> name(v), (a, b) -> Symbol(a, :_, b), cluster)

# Cluster is reffered to a tuple of node interfaces
clusters(factornode::FactorNode) =
    map(factor -> map(i -> @inbounds(interfaces(factornode)[i]), factor), factorisation(factornode))

clusterindex(factornode::FactorNode, v::Symbol)                                = clusterindex(factornode, (v,))
clusterindex(factornode::FactorNode, vindex::Int)                              = clusterindex(factornode, (vindex,))
clusterindex(factornode::FactorNode, vars::NTuple{N, NodeInterface}) where {N} = clusterindex(factornode, map(v -> name(v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Symbol}) where {N}        = clusterindex(factornode, map(v -> interfaceindex(factornode, v), vars))

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

getinterface(factornode::FactorNode, iname::Symbol) =
    @inbounds interfaces(factornode)[interfaceindex(factornode, iname)]

getclusterinterfaces(factornode::FactorNode, cindex::Int) =
    @inbounds map(i -> interfaces(factornode)[i], factorisation(factornode)[cindex])

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
    print(
        io,
        "FactorNodePipeline(functional_dependencies = $(pipeline.functional_dependencies), extra_stages = $(pipeline.extra_stages)"
    )
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

This pipeline translates directly to enforcing a variational message passing scheme. In order to compute a message out of some edge, this pipeline requires
messages from edges within the same edge-cluster and marginals over other edge-clusters.

See also: [`ReactiveMP.RequireMessageFunctionalDependencies`](@ref), [`ReactiveMP.RequireMarginalFunctionalDependencies`](@ref), [`ReactiveMP.RequireEverythingFunctionalDependencies`](@ref)
"""
struct DefaultFunctionalDependencies <: AbstractNodeFunctionalDependenciesPipeline end

function message_dependencies(
    ::DefaultFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
    # First we remove current edge index from the list of dependencies
    vdependencies = TupleTools.deleteat(varcluster, varclusterindex(varcluster, iindex))
    # Second we map interface indices to the actual interfaces
    return map(inds -> map(i -> @inbounds(nodeinterfaces[i]), inds), vdependencies)
end

function marginal_dependencies(
    ::DefaultFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
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

function message_dependencies(
    dependencies::RequireMessageFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)

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
        return message_dependencies(
            DefaultFunctionalDependencies(),
            nodeinterfaces,
            nodelocalmarginals,
            varcluster,
            cindex,
            iindex
        )
    end
end

function marginal_dependencies(
    ::RequireMessageFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
    return marginal_dependencies(
        DefaultFunctionalDependencies(),
        nodeinterfaces,
        nodelocalmarginals,
        varcluster,
        cindex,
        iindex
    )
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

function message_dependencies(
    ::RequireMarginalFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
    return message_dependencies(
        DefaultFunctionalDependencies(),
        nodeinterfaces,
        nodelocalmarginals,
        varcluster,
        cindex,
        iindex
    )
end

function marginal_dependencies(
    dependencies::RequireMarginalFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
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
        default = marginal_dependencies(
            DefaultFunctionalDependencies(),
            nodeinterfaces,
            nodelocalmarginals,
            varcluster,
            cindex,
            iindex
        )
        # Find insertion position (probably might be implemented more efficiently)
        insertafter = sum(first(el) < iindex ? 1 : 0 for el in default; init = 0)
        return TupleTools.insertafter(default, insertafter, (extra_localmarginal,))
    else
        return marginal_dependencies(
            DefaultFunctionalDependencies(),
            nodeinterfaces,
            nodelocalmarginals,
            varcluster,
            cindex,
            iindex
        )
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

function ReactiveMP.message_dependencies(
    ::RequireEverythingFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
    # Return all node interfaces including the edge we are trying to compute a message on
    return nodeinterfaces
end

function ReactiveMP.marginal_dependencies(
    ::RequireEverythingFunctionalDependencies,
    nodeinterfaces,
    nodelocalmarginals,
    varcluster,
    cindex,
    iindex
)
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
        msgs_names      = Val{map(name, messages)}
        msgs_observable = combineLatestUpdates(map(m -> messagein(m), messages), PushNew())
        return msgs_names, msgs_observable
    else
        return nothing, of(nothing)
    end
end

function get_marginals_observable(factornode, marginals)
    if !isempty(marginals)
        marginal_names       = Val{map(name, marginals)}
        marginals_streams    = map(marginal -> getmarginal!(factornode, marginal, IncludeAll()), marginals)
        marginals_observable = combineLatestUpdates(marginals_streams, PushNew())
        return marginal_names, marginals_observable
    else
        return nothing, of(nothing)
    end
end

function activate!(model, factornode::AbstractFactorNode)
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

            mapping =
                let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
                    (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
                end

            vmessageout = vmessageout |> map(AbstractMessage, mapping)
            vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
            vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
            vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

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

function getmarginal!(
    factornode::FactorNode,
    localmarginal::FactorNodeLocalMarginal,
    skip_strategy::MarginalSkipStrategy
)
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
        vtag  = Val{name(localmarginal)}
        meta  = metadata(factornode)

        mapping = MarginalMapping(fform, vtag, msgs_names, marginal_names, meta, factornode)
        # TODO: discontinue operator is needed for loopy belief propagation? Check
        marginalout =
            combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here

        return apply_skip_filter(cmarginal, skip_strategy)
    end
end

## TODO
function conjugate_type end

## make_node

"""
    make_node(node)
    make_node(node, options)

Creates a factor node of a given type and options. See the list of avaialble factor nodes below.

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
    error(
        """
        `$(fform)` is not available as a node in the inference engine. Used in `$(name(first(args))) ~ $(fform)(...)` expression.
        Use `@node` macro to add a custom factor node corresponding to `$(fform)`. See `@node` macro for additional documentation and examples.
        """
    )
end

# end

## macro helpers

import .MacroHelpers

"""
    @node(fformtype, sdtype, interfaces_list)


`@node` macro creates a node for a `fformtype` type object. To obtain a list of available nodes use `?make_node`.

# Arguments

- `fformtype`: Either an existing type identifier, e.g. `Normal` or a function type identifier, e.g. `typeof(+)`
- `sdtype`: Either `Stochastic` or `Deterministic`. Defines the type of the functional relationship
- `interfaces_list`: Defines a fixed list of edges of a factor node, by convention the first element should be `out`. Example: `[ out, mean, variance ]`

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

# List of available nodes

See `?make_node`.

See also: [`make_node`](@ref), [`Stochastic`](@ref), [`Deterministic`](@ref)
"""
macro node(fformtype, sdtype, interfaces_list)
    fbottomtype = MacroHelpers.bottom_type(fformtype)
    fuppertype  = MacroHelpers.upper_type(fformtype)

    @assert sdtype ∈ [:Stochastic, :Deterministic] "Invalid sdtype $(sdtype). Can be either Stochastic or Deterministic."

    @capture(interfaces_list, [interfaces_args__]) ||
        error("Invalid interfaces specification.")

    interfaces = map(interfaces_args) do arg
        if @capture(arg, name_Symbol)
            return (name, [])
        elseif @capture(arg, (name_Symbol, aliases = [aliases__]))
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
    names_indexed          = Expr(:tuple, map(name -> Expr(:call, :(ReactiveMP.indexed_name), name), names)...)

    interface_args        = map(name -> :($name::AbstractVariable), names)
    interface_connections = map(name -> :(ReactiveMP.connect!(node, $(Expr(:quote, name)), $name)), names)

    # Check that all arguments within interface refer to the unique var objects
    non_unique_error_sym = gensym(:non_unique_error_sym)
    non_unique_error_msg = :(
        $non_unique_error_sym =
            (fformtype, names) ->
                """
                Non-unique variables used for the creation of the `$(fformtype)` node, which is disallowed.
                Check creation of the `$(fformtype)` with the `[ $(join(names, ", ")) ]` arguments.
                """
    )
    interface_uniqueness = map(enumerate(names)) do (index, name)
        names_without_current = skipindex(names, index)
        return quote
            if Base.in($(name), ($(names_without_current...),))
                Base.error($(non_unique_error_sym)($fformtype, $names_indexed))
            end
        end
    end

    # Here we create helpers function for GraphPPL.jl interfacing
    # They are used to convert interface names from `where { q = q(x, y)q(z) }` to an equivalent tuple respresentation, e.g. `((1, 2), (3, ))`
    # The general recipe to get a proper index is to call `interface_get_index(Val{ :NodeTypeName }, interface_get_name(Val{ :NodeTypeName }, Val{ :name_expression }))`
    interface_name_getters = map(enumerate(interfaces)) do (index, interface)
        name    = first(interface)
        aliases = last(interface)

        index_name_getter  = :(ReactiveMP.interface_get_index(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$(Expr(:quote, name))}}) = $(index))
        name_symbol_getter = :(ReactiveMP.interface_get_name(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$(Expr(:quote, name))}}) = $(Expr(:quote, name)))
        name_index_getter  = :(ReactiveMP.interface_get_name(::Type{Val{$(Expr(:quote, fbottomtype))}}, ::Type{Val{$index}}) = $(Expr(:quote, name)))

        alias_getters = map(aliases) do alias
            return :(
                ReactiveMP.interface_get_name(
                    ::Type{Val{$(Expr(:quote, fbottomtype))}},
                    ::Type{Val{$(Expr(:quote, alias))}}
                ) = $(Expr(:quote, name))
            )
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
            ReactiveMP.collect_factorisation(::$fuppertype, ::Nothing)                      = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, factorisation::Tuple)           = factorisation
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.FullFactorisation) = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.MeanField)         = $names_splitted_indices
        end

    elseif sdtype === :Deterministic
        quote
            ReactiveMP.collect_factorisation(::$fuppertype, ::Nothing)                      = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, factorisation::Tuple)           = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.FullFactorisation) = ($names_indices,)
            ReactiveMP.collect_factorisation(::$fuppertype, ::ReactiveMP.MeanField)         = ($names_indices,)
        end
    else
        error("Unreachable in @node macro.")
    end

    doctype   = rpad(fbottomtype, 30)
    docsdtype = rpad(sdtype, 15)
    docedges  = string(interfaces_list)
    doc       = """
        $doctype : $docsdtype : $docedges
    """

    res = quote
        ReactiveMP.as_node_functional_form(::$fuppertype) = ReactiveMP.ValidNodeFunctionalForm()

        ReactiveMP.sdtype(::$fuppertype) = (ReactiveMP.$sdtype)()

        ReactiveMP.as_node_symbol(::$fuppertype) = $(QuoteNode(fbottomtype))

        @doc $doc
        function ReactiveMP.make_node(::$fuppertype, options::FactorNodeCreationOptions)
            return ReactiveMP.FactorNode(
                $fbottomtype,
                $names_quoted_tuple,
                ReactiveMP.collect_factorisation($fbottomtype, ReactiveMP.factorisation(options)),
                ReactiveMP.collect_meta($fbottomtype, ReactiveMP.metadata(options)),
                ReactiveMP.collect_pipeline($fbottomtype, ReactiveMP.getpipeline(options))
            )
        end

        function ReactiveMP.make_node(::$fuppertype, options::FactorNodeCreationOptions, $(interface_args...))
            node = ReactiveMP.make_node($fbottomtype, options)
            $(non_unique_error_msg)
            $(interface_uniqueness...)
            $(interface_connections...)
            return node
        end

        $(interface_name_getters...)

        $factorisation_collectors
    end

    return esc(res)
end
