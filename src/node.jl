export NodeInterface, IndexedNodeInterface, name, tag, messageout, messagein
export FactorNode, functionalform, interfaces, factorisation, localmarginals, localmarginalnames, metadata
export iscontain, isfactorised, getinterface
export clusters, clusterindex
export deps, connect!, activate!
export make_node, on_make_node, AutoVar
export ValidNodeFunctionalForm, UndefinedNodeFunctionalForm, as_node_functional_form
export sdtype, Deterministic, Stochastic, isdeterministic, isstochastic
export MeanField, FullFactorisation, collect_factorisation
export @node

using Rocket
using TupleTools

import Base: show
import Base: getindex, setindex!, firstindex, lastindex

## Node traits

struct ValidNodeFunctionalForm end
struct UndefinedNodeFunctionalForm end

as_node_functional_form(::Function) = ValidNodeFunctionalForm()
as_node_functional_form(some)       = UndefinedNodeFunctionalForm()

## Node types

"""
    Deterministic

`Deterministic` object used to parametrize factor node object with determinstic type of relationship between variables.

See also: [`Stochastic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref)
"""
struct Deterministic end

"""
    Stochastic

`Stochastic` object used to parametrize factor node object with stochastic type of relationship between variables.

See also: [`Deterministic`](@ref), [`isdeterministic`](@ref), [`isstochastic`](@ref)
"""
struct Stochastic end

"""
    isdeterministic(node)

Function used to check if factor node object is deterministic or not. Returns true or false.

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isstochastic`](@ref)
"""
function isdeterministic end

"""
    isstochastic(node)

Function used to check if factor node object is stochastic or not. Returns true or false.

See also: [`Deterministic`](@ref), [`Stochastic`](@ref), [`isdeterministic`](@ref)
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

"""
    NodeInterface

`NodeInterface` object represents a single node-variable connection.

See also: [`name`](@ref), [`tag`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
mutable struct NodeInterface
    name               :: Symbol
    m_out              :: LazyObservable{AbstractMessage}
    m_in               :: LazyObservable{AbstractMessage}
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    NodeInterface(name::Symbol) = new(name, lazy(AbstractMessage), lazy(AbstractMessage), nothing, 0)
end

Base.show(io::IO, interface::NodeInterface) = print(io, string("Interface(", name(interface), ")"))

"""
    name(interface)

Returns a name of the interface.

See also: [`NodeInterface`](@ref), [`tag`](@ref)
"""
name(symbol::Symbol)              = symbol
name(interface::NodeInterface)    = name(interface.name)

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
    inbound_portal(interface)

Returns an instance of inbound portal of connected variable for the interface

    See also: [`NodeInterface`](@ref), [`connectvariable!`](@ref), [`connectedvar`](@ref)
"""
inbound_portal(interface::NodeInterface) = inbound_portal(connectedvar(interface))

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

Base.show(io::IO, interface::IndexedNodeInterface) = print(io, string("IndexedInterface(", name(interface), ",", index(interface), ")"))

name(interface::IndexedNodeInterface)  = name(interface.interface)
index(interface::IndexedNodeInterface) = interface.index
tag(interface::IndexedNodeInterface)   = (Val{ name(interface) }(), index(interface))

messageout(interface::IndexedNodeInterface) = messageout(interface.interface)
messagein(interface::IndexedNodeInterface)  = messagein(interface.interface)

connectvariable!(interface::IndexedNodeInterface, variable, index) = connectvariable!(interface.interface, variable, index)
connectedvar(interface::IndexedNodeInterface)                      = connectedvar(interface.interface)
connectedvarindex(interface::IndexedNodeInterface)                 = connectedvarindex(interface.interface)
inbound_portal(interface::IndexedNodeInterface)                    = inbound_portal(interface.interface)

##

function node_interfaces_names end

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

## Generic Factor Node

struct FactorNode{F, I, C, M, A, P} <: AbstractFactorNode
    fform          :: F
    interfaces     :: I
    factorisation  :: C
    localmarginals :: M
    metadata       :: A
    portal         :: P
end

function FactorNode(fform::Type{F}, interfaces::I, factorisation::C, localmarginals::M, metadata::A, portal::P) where { F, I, C, M, A, P }
    return FactorNode{Type{F}, I, C, M, A, P}(fform, interfaces, factorisation, localmarginals, metadata, portal)
end

function FactorNode(fform, varnames::NTuple{N, Symbol}, factorisation, metadata, portal) where N
    interfaces     = map(varname -> NodeInterface(varname), varnames)
    localmarginals = FactorNodeLocalMarginals(varnames, factorisation)
    return FactorNode(fform, interfaces, factorisation, localmarginals, metadata, portal)
end

function Base.show(io::IO, factornode::FactorNode)
    println(io, "FactorNode:")
    println(io, string(" form            : ", functionalform(factornode)))
    println(io, string(" sdtype          : ", sdtype(factornode)))
    println(io, string(" interfaces      : ", interfaces(factornode)))
    println(io, string(" factorisation   : ", factorisation(factornode)))
    println(io, string(" local marginals : ", localmarginalnames(factornode)))
    println(io, string(" metadata        : ", metadata(factornode)))
    println(io, string(" portal          : ", outbound_message_portal(factornode)))
end

functionalform(factornode::FactorNode)          = factornode.fform
sdtype(factornode::FactorNode)                  = sdtype(functionalform(factornode))
interfaces(factornode::FactorNode)              = factornode.interfaces
factorisation(factornode::FactorNode)           = factornode.factorisation
localmarginals(factornode::FactorNode)          = factornode.localmarginals.marginals
localmarginalnames(factornode::FactorNode)      = map(name, localmarginals(factornode))
metadata(factornode::FactorNode)                = factornode.metadata
outbound_message_portal(factornode::FactorNode) = factornode.portal

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

function functional_dependencies(factornode::FactorNode, iname::Symbol)
    return functional_dependencies(factornode, interfaceindex(factornode, iname))
end

function functional_dependencies(factornode::FactorNode, iindex::Int)
    cindex  = clusterindex(factornode, iindex)

    nodeinterfaces     = interfaces(factornode)
    nodeclusters       = factorisation(factornode)
    nodelocalmarginals = localmarginals(factornode)

    varcluster = @inbounds nodeclusters[ cindex ]

    message_dependencies  = map(inds -> map(i -> begin return @inbounds nodeinterfaces[i] end, inds), TupleTools.deleteat(varcluster, varclusterindex(varcluster, iindex)))
    marginal_dependencies = TupleTools.deleteat(nodelocalmarginals, cindex)

    return tuple(message_dependencies...), tuple(marginal_dependencies...)
end

function get_messages_observable(factornode, message_dependencies)
    msgs_names      = nothing
    msgs_observable = of(nothing)

    if length(message_dependencies) !== 0
        msgs_names      = Val{ map(name, message_dependencies) }
        msgs_observable = combineLatest(map(m -> messagein(m), message_dependencies), PushNew())
    end

    return msgs_names, msgs_observable
end

function get_marginals_observable(factornode, marginal_dependencies)
    marginal_names       = nothing
    marginals_observable = of(nothing)

    if length(marginal_dependencies) !== 0 
        marginal_names       = Val{ map(name, marginal_dependencies) }
        marginals_streams    = map(marginal -> getmarginal!(factornode, marginal, IncludeAll()), marginal_dependencies)
        marginals_observable = combineLatestUpdates(marginals_streams, PushNew())
    end

    return marginal_names, marginals_observable
end


# Variational Message Passing
apply_mapping(msgs_observable, marginals_observable, mapping) = (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], mapping)

# Fallback for Belief Propagation
apply_mapping(msgs_observable, marginals_observable::SingleObservable{Nothing}, mapping) = mapping

function activate!(model, factornode::AbstractFactorNode)
    fform = functionalform(factornode)
    meta  = metadata(factornode)

    for (iindex, interface) in enumerate(interfaces(factornode))
        message_dependencies, marginal_dependencies = functional_dependencies(factornode, iindex)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        vtag        = tag(interface)
        vconstraint = constraint(connectedvar(interface))
        
        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew()) # TODO check PushEach
        vmessageout = apply(inbound_portal(interface), factornode, vtag, vmessageout)

        mapping = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
        mapping = apply_mapping(msgs_observable, marginals_observable, mapping)

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply(outbound_message_portal(getoptions(model)), factornode, vtag, vmessageout)
        vmessageout = apply(outbound_message_portal(factornode), factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        set!(messageout(interface), vmessageout |> share_recent())
        set!(messagein(interface), messageout(connectedvar(interface), connectedvarindex(interface)))
    end
end

function setmarginal!(factornode::FactorNode, cname::Symbol, marginal)
    lindex = findnext(lmarginal -> name(lmarginal) === cname, localmarginals(factornode), 1)
    @assert lindex !== nothing "Invalid local marginal id: $cname"
    lmarginal = @inbounds localmarginals(factornode)[ lindex ]
    setmarginal!(getstream(lmarginal), marginal)
end

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

        message_dependencies  = tuple(getclusterinterfaces(factornode, clusterindex)...)
        marginal_dependencies = tuple(TupleTools.deleteat(localmarginals(factornode), clusterindex)...)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        fform       = functionalform(factornode)
        vtag        = Val{ name(localmarginal) }
        meta        = metadata(factornode)

        mapping = let fform = fform, vtag = vtag, msgs_names = msgs_names, marginal_names = marginal_names, meta = meta, factornode = factornode
            (dependencies) -> begin 
                messages  = dependencies[1]
                marginals = getrecent(dependencies[2])

                # Marginal is clamped if all of the inputs are clamped
                is_marginal_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

                # Marginal is initial if it is not clamped and all of the inputs are either clamped or initial
                is_marginal_initial = !is_marginal_clamped && (__check_all(m -> is_clamped(m) || is_initial(m), messages) && __check_all(m -> is_clamped(m) || is_initial(m), marginals))

                return Marginal(marginalrule(fform, vtag, msgs_names, messages, marginal_names, marginals, meta, factornode), is_marginal_clamped, is_marginal_initial)
            end
        end

        # TODO: discontinue operater is needed for loopy belief propagation? Check
        marginalout = combineLatest((msgs_observable, marginals_observable), PushNew()) |> discontinue() |> map(Marginal, mapping)

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here

        setstream!(localmarginal, cmarginal)

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
        ReactiveMP.node_interfaces_names(::$fuppertype) = $names_quoted_tuple

        ReactiveMP.as_node_functional_form(::$fuppertype) = ReactiveMP.ValidNodeFunctionalForm()

        ReactiveMP.sdtype(::$fuppertype) = (ReactiveMP.$sdtype)()
        
        function ReactiveMP.make_node(::$fuppertype; factorisation = ($names_indices, ), meta = nothing, portal = ReactiveMP.EmptyPortal())
            return ReactiveMP.FactorNode($fbottomtype, $names_quoted_tuple, ReactiveMP.collect_factorisation($fbottomtype, factorisation), ReactiveMP.collect_meta($fbottomtype, meta), portal)
        end
        
        function ReactiveMP.make_node(::$fuppertype, $(interface_args...); factorisation = ($names_indices, ), meta = nothing, portal = ReactiveMP.EmptyPortal())
            node = ReactiveMP.make_node($fbottomtype, factorisation = factorisation, meta = meta, portal = portal)
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