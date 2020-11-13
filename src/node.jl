export NodeInterface, name, messageout, messagein
export FactorNode, functionalform, variables, factorisation, factors, varindex, iscontain, isfactorised, getvariable
export getcluster, clusters, clusterindex
export deps, connect!, activate!
export make_node, AutoVar
export Marginalisation
export sdtype, Deterministic, Stochastic, isdeterministic, isstochastic
export MeanField, FullFactorisation
export @node

using Rocket

import Base: show
import Base: getindex, setindex!, firstindex, lastindex

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
    name  :: Symbol
    props :: FactorNodeLocalMarginalProps

    FactorNodeLocalMarginal(name::Symbol) = new(name, FactorNodeLocalMarginalProps())
end

name(localmarginal::FactorNodeLocalMarginal) = localmarginal.name

getstream(localmarginal::FactorNodeLocalMarginal) = localmarginal.props.stream
setstream!(localmarginal::FactorNodeLocalMarginal, observable::MarginalObservable) = localmarginal.props.stream = observable

"""
    FactorNodeLocalMarginals

This object acts as an iterable and indexable proxy for local marginals for some node. 
"""
struct FactorNodeLocalMarginals{M}
    marginals :: M
end

function FactorNodeLocalMarginals(interfaces, factorisation)
    marginals  = map(cname -> FactorNodeLocalMarginal(cname), clusternames(interfaces, factorisation))
    return FactorNodeLocalMarginals(marginals)
end

@inline function __findindex(localmarginals::FactorNodeLocalMarginals, cname::Symbol)
    index = findnext(lmarginal -> name(lmarginal) === cname, localmarginals.marginals, 1)
    return index !== nothing ? index : throw("Invalid local marginal id: $s")
end

Base.getindex(localmarginals::FactorNodeLocalMarginals, cname::Symbol)              = @inbounds getstream(localmarginals.marginals[__findindex(localmarginals, cname)])
Base.setindex!(localmarginals::FactorNodeLocalMarginals, observable, cname::Symbol) = @inbounds setstream!(localmarginals.marginals[__findindex(localmarginals, cname)], observable)

Base.firstindex(localmarginals::FactorNodeLocalMarginals) = firstindex(localmarginals.marginals)
Base.lastindex(localmarginals::FactorNodeLocalMarginals)  = lastindex(localmarginals.marginals)

## FactorNode

struct FactorNode{F, T, I, C, M, A}
    fform          :: F
    sdtype         :: T
    interfaces     :: I
    factorisation  :: C
    localmarginals :: M
    metadata       :: A
end

function FactorNode(fform::Type{F}, sdtype::T, interfaces::I, factorisation::C, localmarginals::M, metadata::A) where { F, T, I, C, M, A }
    return FactorNode{Type{F}, T, I, C, M, A}(fform, sdtype, interfaces, factorisation, localmarginals, metadata)
end

function FactorNode(fform, sdtype, variables::NTuple{N, Symbol}, factorisation, metadata) where N
    interfaces     = map(variable -> NodeInterface(variable), variables)
    localmarginals = FactorNodeLocalMarginals(variables, factorisation)
    return FactorNode(fform, sdtype, interfaces, factorisation, localmarginals, metadata)
end

function Base.show(io::IO, factornode::FactorNode)
    println(io, "FactorNode:")
    println(io, string(" form            : ", functionalform(factornode)))
    println(io, string(" sdtype          : ", sdtype(factornode)))
    println(io, string(" interfaces      : ", interfaces(factornode)))
    println(io, string(" factorisation   : ", factorisation(factornode)))
    println(io, string(" local marginals : ", name.(factornode.localmarginals.marginals)))
    println(io, string(" metadata        : ", metadata(factornode)))
end

functionalform(factornode::FactorNode) = factornode.fform
sdtype(factornode::FactorNode)         = factornode.sdtype
interfaces(factornode::FactorNode)     = factornode.interfaces
factorisation(factornode::FactorNode)  = factornode.factorisation
localmarginals(factornode::FactorNode) = factornode.localmarginals
metadata(factornode::FactorNode)       = factornode.metadata

isstochastic(factornode::FactorNode)    = isstochastic(sdtype(factornode))
isdeterministic(factornode::FactorNode) = isdeterministic(sdtype(factornode))

clustername(cluster) = mapreduce(v -> name(v), (a, b) -> Symbol(a, :_, b), cluster)

clusternames(factornode::FactorNode)                              = map(clustername, clusters(factornode))
clusternames(variables::NTuple{N, Symbol}, factorisation) where N = map(clustername, map(q -> map(v -> variables[v], q), factorisation))

getcluster(factornode::FactorNode, i)                = @inbounds factornode.factorisation[i]
clusters(factornode::FactorNode)                     = map(factor -> map(i -> begin return @inbounds factornode.variables[i] end, factor), factorisation(factornode))
clusterindex(factornode::FactorNode, v::Symbol)      = clusterindex(factornode, varindex(factornode, v))
clusterindex(factornode::FactorNode, vindex::Int)    = findfirst(cluster -> vindex ∈ cluster, factorisation(factornode))

clusterindex(factornode::FactorNode, vars::NTuple{N, NodeInterface}) where N = clusterindex(factornode, map(v -> name(v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Symbol}) where N       = clusterindex(factornode, map(v -> varindex(factornode, v), vars))
clusterindex(factornode::FactorNode, vars::NTuple{N, Int}) where N          = findfirst(cluster -> all(v -> v ∈ cluster, vars), factorisation(factornode))

varclusterindex(cluster, vindex::Int) = findfirst(index -> index === vindex, cluster)

function getvariable(factornode::FactorNode, v::Symbol)
    vindex = varindex(factornode, v)
    @assert vindex !== nothing
    return @inbounds interfaces(factornode)[vindex]
end

varindex(factornode::FactorNode, v::Symbol)    = findfirst(d -> d === v, map(v -> name(v), interfaces(factornode)))
iscontain(factornode::FactorNode, v::Symbol)   = varindex(factornode, v) !== nothing
isfactorised(factornode::FactorNode, f)        = findfirst(d -> d == f, factorisation(factornode)) !== nothing

function connect!(factornode::FactorNode, v::Symbol, variable) 
    return connect!(factornode::FactorNode, v::Symbol, variable, getlastindex(variable))
end

function connect!(factornode::FactorNode, v::Symbol, variable, index)
    vindex = varindex(factornode, v)

    @assert vindex !== nothing

    nodeinterfaces = interfaces(factornode)
    varinterface   = @inbounds nodeinterfaces[vindex]

    connectvariable!(varinterface, variable, index)
    setmessagein!(variable, index, messageout(varinterface))
end

function deps(factornode::FactorNode, v::Symbol)
    vindex = varindex(factornode, v)
    cindex = clusterindex(factornode, vindex)

    @assert vindex !== nothing
    @assert cindex !== nothing

    vars = interfaces(factornode)
    cls  = factorisation(factornode)

    factor  = @inbounds cls[cindex]
    vcindex = varclusterindex(factor, vindex)

    @assert vcindex !== nothing

    # TODO Consider to change this line with map/map
    # TODO benchmark it
    mdeps       = map(inds -> map(i -> begin return @inbounds vars[i] end, inds), skipindex(factor, vcindex))
    clusterdeps = map(inds -> map(i -> begin return @inbounds vars[i] end, inds), skipindex(cls, cindex))

    return mdeps, clusterdeps
end

function activate!(model, factornode::FactorNode)
    for variable in interfaces(factornode)
        mdeps, clusterdeps = deps(factornode, name(variable))

        msgs_names      = nothing
        msgs_observable = of(nothing)

        cluster_names       = nothing
        clusters_observable = of(nothing)

        if length(mdeps) !== 0
            msgs_names      = Val{ tuple(name.(mdeps)...) }
            msgs_observable = combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew())
        end

        if length(clusterdeps) !== 0 
            cluster_names       = Val{ tuple(clustername.(clusterdeps)...) }
            clusters_observable = combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushNew())
        end

        gate        = message_gate(model)
        fform       = functionalform(factornode)
        vtag        = tag(variable)
        vconstraint = Marginalisation()
        meta        = metadata(factornode)
        mapping     = switch_map(Message, (d) -> begin
            message = rule(fform, vtag, vconstraint, msgs_names, d[1], cluster_names, d[2], meta, factornode)
            return cast_to_subscribable(message) |> map(Message, (d) -> as_message(gate!(gate, factornode, variable, d)))
        end) 
         
        vmessageout = apply(message_out_transformer(model), combineLatest(msgs_observable, clusters_observable, strategy = PushEach()))

        set!(messageout(variable), vmessageout |> mapping |> share_replay(1))
        set!(messagein(variable), messageout(connectedvar(variable), connectedvarindex(variable)))
    end
end

function setmarginal!(factornode::FactorNode, name::Symbol, v)
    marginal = factornode.localmarginals[name]
    if marginal === nothing
        throw("Marginal with name $(name) does not exist on factor node $(factornode)")
    end
    setmarginal!(marginal, v)
end

function getmarginal!(factornode::FactorNode, cluster)
    cname = clustername(cluster)

    if factornode.localmarginals[cname] !== nothing
        return factornode.localmarginals[cname]
    end

    if length(cluster) === 1 # Cluster contains only one variable, we can take marginal over this variable
        vmarginal = getmarginal(connectedvar(cluster[1]))
        factornode.localmarginals[cname] = vmarginal
        return vmarginal
    else
        cmarginal = MarginalObservable()
        factornode.localmarginals[cname] = cmarginal
        # TODO generalise as a separate function
        mdeps = cluster

        vars = interfaces(factornode)
        cls  = factorisation(factornode)

        cindex      = clusterindex(factornode, cluster)
        clusterdeps = map(inds -> map(i -> vars[i], inds), skipindex(cls, cindex))

        msgs_names      = nothing
        msgs_observable = of(nothing)

        cluster_names       = nothing
        clusters_observable = of(nothing)

        if length(mdeps) !== 0
            msgs_names      = Val{ tuple(name.(mdeps)...) }
            msgs_observable = combineLatest(map(m -> messagein(m), mdeps)..., strategy = PushNew())
        end

        if length(clusterdeps) !== 0 
            cluster_names       = Val{ tuple(clustername.(clusterdeps)...) }
            clusters_observable = combineLatest(map(c -> getmarginal!(factornode, c), clusterdeps)..., strategy = PushNew())
        end

        fform       = functionalform(factornode)
        vtag        = Val{ clustername(cluster) }
        meta        = metadata(factornode)
        mapping     = map(Marginal, (d) -> as_marginal(marginalrule(fform, vtag, msgs_names, d[1], cluster_names, d[2], meta, factornode)))
        marginalout = combineLatest(msgs_observable, clusters_observable, strategy = PushEach()) |> discontinue() |> mapping

        connect!(cmarginal, marginalout |> share_replay(1))

        return cmarginal
    end

    throw("Unsupported marginal size: $(length(cluster))")
end

## make_node

function make_node end

function interface_get_index end
function interface_get_name end

## AutoVar

struct AutoVar
    name :: Symbol
end

getname(autovar::AutoVar) = autovar.name

function make_node(fform, autovar::AutoVar, args...; kwargs...)
    var  = randomvar(getname(autovar))
    node = make_node(fform, var, args...; kwargs...)
    return node, var
end

# TODO: extend this case for more cases
function make_node(fform::Function, autovar::AutoVar, inputs::Vararg{ <: ConstVariable{ <: Dirac } }; kwargs...)
    var  = constvar(getname(autovar), fform(map((d) -> getpointmass(getconstant(d)), inputs)...))
    return nothing, var
end

# TODO: This can intersect with T = Distributions, what to do?
# function make_node(::Type{ T }, autovar::AutoVar, inputs::Vararg{ <: ConstVariable{ <: Dirac } }; kwargs...) where T
#     var  = constvar(getname(autovar), T(map((d) -> getpointmass(getconstant(d)), inputs)...))
#     return nothing, var
# end

# TODO
# function make_node(fform::Function, autovar::AutoVar, inputs::Vararg{ <: Union{ <: ConstVariable{ <: Dirac }, <: DataVariable{ <: Any, <: Dirac } } })
    # combineLatest + map
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
        collect_factorisation(::$formtype, ::FullFactorisation) = ($names_indices, )
        collect_factorisation(::$formtype, ::MeanField) = $names_splitted_indices
    end
    
    res = quote
        
        function ReactiveMP.make_node(::$formtype; factorisation = ($names_indices, ), meta = nothing)
            return FactorNode($form, $sdtype, $names_quoted_tuple, collect_factorisation($form, factorisation), meta)
        end
        
        function ReactiveMP.make_node(::$formtype, $(interface_args...); factorisation = ($names_indices, ), meta = nothing)
            node = make_node($form, factorisation = factorisation, meta = meta)
            $(interface_connections...)
            return node
        end

        $(interface_name_getters...)
        $factorisation_collectors

    end
    
    return esc(res)
end