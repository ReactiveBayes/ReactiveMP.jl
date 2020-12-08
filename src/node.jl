export NodeInterface, name, messageout, messagein
export FactorNode, functionalform, interfaces, factorisation, locamarginals, localmarginalnames
export iscontain, isfactorised, getinterface
export clusters, clusterindex
export deps, connect!, activate!
export make_node, on_make_node, AutoVar
export Marginalisation
export ValidNodeFunctionalForm, UndefinedNodeFunctionalForm, as_node_functional_form
export sdtype, Deterministic, Stochastic, isdeterministic, isstochastic
export MeanField, FullFactorisation, collect_factorisation
export @node

using Rocket

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

# Examples 

```julia

using ReactiveMP

f = ReactiveMP.collect_factorisation(NormalMeanVariance, MeanField()) # ((1,), (2, ), (3, ))
```

See also: [`MeanField`](@ref), [`FullFactorisation`](@ref)
"""
function collect_factorisation end

collect_factorisation(::Any, factorisation::Tuple) = factorisation

## Variable constraints

struct Marginalisation end

## NodeInterface Props

mutable struct NodeInterfaceProps
    connected_variable :: Union{Nothing, AbstractVariable}
    connected_index    :: Int

    NodeInterfaceProps() = new(nothing, 0)
end

## NodeInterface

"""
    NodeInterface

`NodeInterface` object represents a single node-variable connection.

See also: [`name`](@ref), [`tag`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
struct NodeInterface
    name  :: Symbol
    m_out :: LazyObservable{Message}
    m_in  :: LazyObservable{Message}
    props :: NodeInterfaceProps

    NodeInterface(name::Symbol) = new(name, lazy(Message), lazy(Message), NodeInterfaceProps())
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

See also: [`NodeInterface`](@ref), [`name`](@ref), [`activate!`](@ref)
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
    interface.props.connected_variable = variable
    interface.props.connected_index    = index
end

"""
    connectedvar(interface)

Returns connected variable for the interface.

See also: [`NodeInterface`](@ref), [`connectvariable!`](@ref), [`connectedvarindex`](@ref)
"""
connectedvar(interface::NodeInterface)      = interface.props.connected_variable

"""
    connectedvarindex(interface)

Returns an index of connected variable for the interface.

See also: [`NodeInterface`](@ref), [`connectvariable!`](@ref), [`connectedvar`](@ref)
"""
connectedvarindex(interface::NodeInterface) = interface.props.connected_index

## IndexedNodeInterface
## Used for dynamic number of inputs
struct IndexedNodeInterface
    index     :: Int
    interface :: NodeInterface
end

Base.show(io::IO, interface::IndexedNodeInterface) = print(io, string("IndexedInterface(", name(interface), ",", index(interface), ")"))

name(interface::IndexedNodeInterface)  = name(interface.interface)
index(interface::IndexedNodeInterface) = interface.index
tag(interface::IndexedNodeInterface)   = (Val{ name(interface) }, Val{ index(interface) })

messageout(interface::IndexedNodeInterface) = messageout(interface.interface)
messagein(interface::IndexedNodeInterface)  = messagein(interface.interface)

connectvariable!(interface::IndexedNodeInterface, variable, index) = connectvariable!(interface.interface, variable, index)
connectedvar(interface::IndexedNodeInterface)                      = connectedvar(interface.interface)
connectedvarindex(interface::IndexedNodeInterface)                 = connectedvarindex(interface.interface)

## FactorNodeLocalMarginals

mutable struct FactorNodeLocalMarginalProps
    stream :: Union{Nothing, MarginalObservable}

    FactorNodeLocalMarginalProps() = new(nothing)
end

"""
    FactorNodeLocalMarginal

# Fields
1. name  :: Symbol - name of local marginal, e.g. `μ`. Name is `_` separated in case of joint, eg. `μ_τ`
2. props :: FactorNodeLocalMarginalProps - mutable object which is holding a stream of marginals or nothing

This object represents local marginals for some specific factor node. 
Local marginal can be joint in case of structured factorisation. 
Local to factor node marginal also can be shared with a corresponding marginal of some random variable.

See also: [`FactorNodeLocalMarginals`](@ref)
"""
struct FactorNodeLocalMarginal 
    index :: Int
    name  :: Symbol
    props :: FactorNodeLocalMarginalProps

    FactorNodeLocalMarginal(index::Int, name::Symbol) = new(index, name, FactorNodeLocalMarginalProps())
end

index(localmarginal::FactorNodeLocalMarginal) = localmarginal.index
name(localmarginal::FactorNodeLocalMarginal)  = localmarginal.name

getstream(localmarginal::FactorNodeLocalMarginal) = localmarginal.props.stream
setstream!(localmarginal::FactorNodeLocalMarginal, observable::MarginalObservable) = localmarginal.props.stream = observable

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

## FactorNode

abstract type AbstractFactorNode end

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

isstochastic(factornode::FactorNode)    = isstochastic(sdtype(factornode))
isdeterministic(factornode::FactorNode) = isdeterministic(sdtype(factornode))

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

    message_dependencies  = map(inds -> map(i -> begin return @inbounds nodeinterfaces[i] end, inds), skipindex(varcluster, varclusterindex(varcluster, iindex)))
    marginal_dependencies = skipindex(nodelocalmarginals, cindex)

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
        marginals_observable = combineLatest(map(marginal -> getmarginal!(factornode, marginal), marginal_dependencies), PushNew())
    end

    return marginal_names, marginals_observable
end

function activate!(model, factornode::AbstractFactorNode)
    for (iindex, interface) in enumerate(interfaces(factornode))
        message_dependencies, marginal_dependencies = functional_dependencies(factornode, iindex)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = Marginalisation()
        meta        = metadata(factornode)
        
        vmessageout = combineLatest(msgs_observable, marginals_observable, strategy = PushEach())

        vmessageout = vmessageout |> switch_map(Message, (d) -> begin 
            return cast_to_message_subscribable(rule(fform, vtag, vconstraint, msgs_names, d[1], marginal_names, d[2], meta, factornode))
        end)

        vmessageout = apply(outbound_message_portal(getoptions(model)), factornode, vtag, vmessageout)
        vmessageout = apply(outbound_message_portal(factornode), factornode, vtag, vmessageout)

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
    cached_stream = getstream(localmarginal)

    if cached_stream !== nothing
        return cached_stream
    end

    clusterindex = index(localmarginal)

    marginalname = name(localmarginal)
    marginalsize = @inbounds length(factorisation(factornode)[ clusterindex ])

    if marginalsize === 1 
        # Cluster contains only one variable, we can take marginal over this variable
        vmarginal = getmarginal(connectedvar(getinterface(factornode, marginalname)))
        setstream!(localmarginal, vmarginal)
        return vmarginal
    else
        cmarginal = MarginalObservable()

        message_dependencies  = tuple(getclusterinterfaces(factornode, clusterindex)...)
        marginal_dependencies = tuple(skipindex(localmarginals(factornode), clusterindex)...)

        msgs_names, msgs_observable          = get_messages_observable(factornode, message_dependencies)
        marginal_names, marginals_observable = get_marginals_observable(factornode, marginal_dependencies)

        fform       = functionalform(factornode)
        vtag        = Val{ name(localmarginal) }
        meta        = metadata(factornode)
        mapping     = map(Marginal, (d) -> as_marginal(marginalrule(fform, vtag, msgs_names, d[1], marginal_names, d[2], meta, factornode)))
        marginalout = combineLatest(msgs_observable, marginals_observable, strategy = PushEach()) |> mapping

        connect!(cmarginal, marginalout) # MarginalObservable has RecentSubject by default, there is no need to share_recent() here

        setstream!(localmarginal, cmarginal)

        return cmarginal
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

function make_node(fform::Function, autovar::AutoVar, args::Vararg{ <: DataVariable{ <: Dirac } }; kwargs...)
    subject = combineLatest(tuple(map((a) -> messageout(a, getlastindex(a)) |> map(Any, getdata), args)...), PushNew()) |> map(Message, (d...) -> as_message(fform(d...)))
    var     = datavar(getname(autovar), Any, subject = subject)
    return nothing, var
end

# end

## macro helpers

macro node(fformtype, fsdtype, finterfaces)

    form       = __extract_fform_macro_rule(fformtype)
    formtype   = __extract_fformtype_macro_rule(fformtype)
    sdtype     = __extract_sdtype_macro_rule(fsdtype)
    interfaces = __extract_interfaces_macro_rule(finterfaces)
    
    names = map(d -> d[:name], interfaces)
    
    names_quoted_tuple     = Expr(:tuple, map(name -> Expr(:quote, name), names)...)
    names_indices          = Expr(:tuple, map(i -> i, 1:length(names))...)
    names_splitted_indices = Expr(:tuple, map(i -> Expr(:tuple, i), 1:length(names))...)
    
    interface_args        = map(name -> :($name::AbstractVariable), names)
    interface_connections = map(name -> :(connect!(node, $(Expr(:quote, name)), $name)), names)

    interface_name_getters = map(enumerate(interfaces)) do (index, interface)
        name    = interface[:name]
        aliases = interface[:aliases]

        index_name_getter  = :(ReactiveMP.interface_get_index(::Type{ Val{ $(Expr(:quote, form)) } }, ::Type{ Val{ $(Expr(:quote, name)) } }) = $(index))
        name_symbol_getter = :(ReactiveMP.interface_get_name(::Type{ Val{ $(Expr(:quote, form)) } }, ::Type{ Val{ $(Expr(:quote, name)) } }) = $(Expr(:quote, name)))
        name_index_getter  = :(ReactiveMP.interface_get_name(::Type{ Val{ $(Expr(:quote, form)) } }, ::Type{ Val{ $index } }) = $(Expr(:quote, name)))

        alias_getters = map(aliases) do alias
            return :(ReactiveMP.interface_get_name(::Type{ Val{ $(Expr(:quote, form)) } }, ::Type{ Val{ $(Expr(:quote, alias)) } }) = $(Expr(:quote, name)))
        end
    
        return quote
            $index_name_getter
            $name_symbol_getter
            $name_index_getter
            $(alias_getters...)
        end
    end

    factorisation_collectors = quote
        ReactiveMP.collect_factorisation(::$formtype, ::FullFactorisation) = ($names_indices, )
        ReactiveMP.collect_factorisation(::$formtype, ::MeanField) = $names_splitted_indices
    end

    make_node_const_mapping = if sdtype === :Stochastic
        quote
            function ReactiveMP.make_node(fform::$formtype, autovar::AutoVar, args::Vararg{ <: ConstVariable{ <: Dirac } }; kwargs...)
                var  = randomvar(getname(autovar))
                node = make_node(fform, var, args...; kwargs...)
                return node, var
            end
        end
    elseif sdtype === :Deterministic
        quote
            function ReactiveMP.make_node(fform::$formtype, autovar::AutoVar, args::Vararg{ <: ConstVariable{ <: Dirac } }; kwargs...)
                var  = constvar(getname(autovar), fform(map((d) -> getconst(d), args)...))
                return nothing, var
            end
        end
    end
    
    res = quote

        ReactiveMP.as_node_functional_form(::$formtype) = ValidNodeFunctionalForm()

        ReactiveMP.sdtype(::$formtype) = ($sdtype)()
        
        function ReactiveMP.make_node(::$formtype; factorisation = ($names_indices, ), meta = nothing, portal = EmptyPortal())
            return FactorNode($form, $names_quoted_tuple, collect_factorisation($form, factorisation), meta, portal)
        end
        
        function ReactiveMP.make_node(::$formtype, $(interface_args...); factorisation = ($names_indices, ), meta = nothing, portal = EmptyPortal())
            node = make_node($form, factorisation = factorisation, meta = meta, portal = portal)
            $(interface_connections...)
            return node
        end

        function ReactiveMP.make_node(fform::$formtype, autovar::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...)
            var  = randomvar(getname(autovar))
            node = make_node(fform, var, args...; kwargs...)
            return node, var
        end

        $(make_node_const_mapping)
        $(interface_name_getters...)
        $factorisation_collectors

    end
    
    return esc(res)
end