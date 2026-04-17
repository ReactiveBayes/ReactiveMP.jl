"""
    ReactiveMP.NodeInterface

Represents a single directed connection between a factor node and an [`ReactiveMP.AbstractVariable`](@ref).

Each interface owns one [`ReactiveMP.MessageObservable`](@ref) (`m_out`) — the *outbound* message stream from this node toward the connected variable. The constructor immediately calls [`ReactiveMP.create_new_stream_of_inbound_messages!`](@ref) on the variable, which allocates a per-connection slot in the variable's `input_messages` and returns the same observable together with its index. This means `m_out` for the interface is the inbound message stream from the variable's perspective.

After graph construction the streams are unconnected (lazy). [`ReactiveMP.activate!`](@ref) wires `m_out` to the result of the message update rule via [`ReactiveMP.set_stream_of_outbound_messages!`](@ref).

See also: [`ReactiveMP.IndexedNodeInterface`](@ref), [`ReactiveMP.get_stream_of_outbound_messages`](@ref), [`ReactiveMP.get_stream_of_inbound_messages`](@ref)
"""
struct NodeInterface
    name::Symbol
    m_out::MessageObservable{AbstractMessage}
    variable::AbstractVariable
    message_index::Int

    function NodeInterface(name::Symbol, variable::AbstractVariable)
        # `inbound message` for variable is `m_out` for the interface
        m_out, message_index = create_new_stream_of_inbound_messages!(variable)
        return new(name, m_out, variable, message_index)
    end

    function NodeInterface(name::Symbol, variable::Any)
        return NodeInterface(name, convert(AbstractVariable, variable))
    end
end

Base.show(io::IO, interface::NodeInterface) = print(
    io, "Interface(", name(interface), ")"
)

israndom(interface::NodeInterface) = israndom(interface.variable)
isdata(interface::NodeInterface)   = isdata(interface.variable)
isconst(interface::NodeInterface)  = isconst(interface.variable)

"""
    name(interface)

Returns a name of the interface.
"""
name(symbol::Symbol) = symbol
name(interface::NodeInterface) = name(interface.name)

"""
    tag(interface)

Returns a tag of the interface in the form of `Val{ name(interface) }`.
The major difference between tag and name is that it is possible to dispath on interface's tag in message computation rule.
"""
tag(interface::NodeInterface) = Val{name(interface)}()

"""
    get_stream_of_outbound_messages(interface)

Returns an outbound messages stream from the given interface.
"""
get_stream_of_outbound_messages(interface::NodeInterface) = interface.m_out

"""
    ReactiveMP.set_stream_of_outbound_messages!(interface, stream)

Connects `stream` to the outbound message observable of `interface`.
See also [`ReactiveMP.get_stream_of_outbound_messages`](@ref), [`ReactiveMP.get_stream_of_inbound_messages`](@ref).
"""
set_stream_of_outbound_messages!(interface::NodeInterface, stream) = connect!(
    get_stream_of_outbound_messages(interface), stream
)

"""
    get_stream_of_inbound_messages(interface)

Returns an inbound messages stream from the given interface.
"""
get_stream_of_inbound_messages(interface::NodeInterface) = get_stream_of_outbound_messages(
    interface.variable, interface.message_index
)

"""
    getvariable(interface)

Returns a variable connected to the given interface.
"""
getvariable(interface::NodeInterface) = interface.variable

"""
    ReactiveMP.IndexedNodeInterface

A thin wrapper around [`ReactiveMP.NodeInterface`](@ref) that adds a positional `index`, used for nodes with a variable-length list of same-named edges (e.g. the `means` or `precisions` of a Gaussian Mixture node). All stream and variable accessors delegate to the wrapped interface.

See also: [`ReactiveMP.NodeInterface`](@ref), [`ReactiveMP.ManyOf`](@ref)
"""
struct IndexedNodeInterface
    index     :: Int
    interface :: NodeInterface
end

Base.show(io::IO, interface::IndexedNodeInterface) = print(
    io,
    string("IndexedInterface(", index(interface), ", ", name(interface), ")"),
)

index(interface::IndexedNodeInterface) = interface.index
name(interface::IndexedNodeInterface)  = name(interface.interface)
tag(interface::IndexedNodeInterface)   = (tag(interface.interface), index(interface))

get_stream_of_outbound_messages(interface::IndexedNodeInterface) = get_stream_of_outbound_messages(
    interface.interface
)
set_stream_of_outbound_messages!(interface::IndexedNodeInterface, stream) = set_stream_of_outbound_messages!(
    interface.interface, stream
)
get_stream_of_inbound_messages(interface::IndexedNodeInterface) = get_stream_of_inbound_messages(
    interface.interface
)
getvariable(interface::IndexedNodeInterface) = getvariable(interface.interface)

israndom(interface::IndexedNodeInterface) = israndom(interface.interface)
isdata(interface::IndexedNodeInterface)   = isdata(interface.interface)
isconst(interface::IndexedNodeInterface)  = isconst(interface.interface)

"""
Some nodes use `IndexedInterface`, `ManyOf` structure reflects a collection of marginals from the collection of `IndexedInterface`s. `@rule` macro 
also treats `ManyOf` specially.
"""
struct ManyOf{T}
    collection::T
end

Base.show(io::IO, manyof::ManyOf) = print(
    io, "ManyOf(", join(manyof.collection, ",", ""), ")"
)

Rocket.getrecent(many::ManyOf) = ManyOf(getrecent(many.collection))

getdata(many::ManyOf)    = getdata(many.collection)
is_clamped(many::ManyOf) = is_clamped(many.collection)
is_initial(many::ManyOf) = is_initial(many.collection)
typeofdata(many::ManyOf) = typeof(ManyOf(many.collection))

paramfloattype(many::ManyOf) = paramfloattype(many.collection)

rule_method_error_type_nameof(::Type{T}) where {V, T <: ManyOf{V}} = begin
    # V is the tuple type carried in ManyOf{V}
    fts = fieldtypes(V)               # get the element type tuple, works for NTuple and Tuple
    N = length(fts)                   # number of elements

    if N == 0
        return "ManyOf{0, }"
    end

    # If all element types are the same, produce the compact NTuple-style message:
    if all(ft -> ft === fts[1], fts)
        elt = dropproxytype(fts[1])
        return string(
            "ManyOf{", N, ", ", rule_method_error_type_nameof(elt), "}"
        )
    end

    # Otherwise produce the Union of element type names:
    unions = join(
        map(r -> rule_method_error_type_nameof(dropproxytype(r)), fts), ","
    )
    return string("ManyOf{", N, ", Union{", unions, "}}")
end

Base.iterate(many::ManyOf)        = iterate(many.collection)
Base.iterate(many::ManyOf, state) = iterate(many.collection, state)

Base.length(many::ManyOf) = length(many.collection)

struct ManyOfObservable{S} <: Subscribable{ManyOf}
    source::S
end

Rocket.getrecent(observable::ManyOfObservable) = ManyOf(
    Rocket.getrecent(observable.source)
)

@inline function Rocket.on_subscribe!(observable::ManyOfObservable, actor)
    return subscribe!(observable.source |> map(ManyOf, (d) -> ManyOf(d)), actor)
end

function combineLatestMessagesInUpdates(
    indexed::NTuple{N, <:IndexedNodeInterface}
) where {N}
    return ManyOfObservable(
        combineLatestUpdates(
            map((in) -> get_stream_of_inbound_messages(in), indexed), PushNew()
        ),
    )
end
