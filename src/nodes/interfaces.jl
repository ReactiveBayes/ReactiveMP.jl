"""
    NodeInterface

`NodeInterface` object represents a single node-variable connection.

See also: [`name`](@ref), [`tag`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
struct NodeInterface
    name::Symbol
    m_out::MessageObservable{AbstractMessage}
    variable::AbstractVariable
    message_index::Int

    function NodeInterface(name::Symbol, variable::AbstractVariable)
        # `messagein` for variable is `m_out` for the interface
        m_out, message_index = create_messagein!(variable)
        return new(name, m_out, variable, message_index)
    end
end

Base.show(io::IO, interface::NodeInterface) = print(io, string("Interface(", name(interface), ")"))

israndom(interface::NodeInterface) = israndom(interface.variable)
isdata(interface::NodeInterface)   = isdata(interface.variable)
isconst(interface::NodeInterface)  = isconst(interface.variable)

"""
    name(interface)

Returns a name of the interface.

See also: [`NodeInterface`](@ref), [`tag`](@ref)
"""
name(symbol::Symbol) = symbol
name(interface::NodeInterface) = name(interface.name)

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
messagein(interface::NodeInterface) = messageout(interface.variable, interface.message_index)

"""
    getvariable(interface)

Returns a variable connected to the given interface.

See also: [`NodeInterface`](@ref), [`messageout`](@ref), [`messagein`](@ref)
"""
getvariable(interface::NodeInterface) = interface.variable

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

messageout(interface::IndexedNodeInterface) = error("TODO") # messageout(interface.interface)
messagein(interface::IndexedNodeInterface)  = error("TODO") # messagein(interface.interface)

connectvariable!(interface::IndexedNodeInterface, properties, index) = error("TODO") # connectvariable!(interface.interface, properties, index)
connected_properties(interface::IndexedNodeInterface) = error("TODO") # connected_properties(interface.interface)
connectedvarindex(interface::IndexedNodeInterface) = error("TODO") # connectedvarindex(interface.interface)
get_pipeline_stages(interface::IndexedNodeInterface) = error("TODO") # get_pipeline_stages(interface.interface)

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