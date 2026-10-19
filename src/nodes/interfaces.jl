"""
    ReactiveMP.NodeInterface(name::Symbol, variable)

One edge of a factor node: the interface `name` and the variable it connects to. Creating it
calls [`ReactiveMP.create_new_stream_of_inbound_messages!`](@ref) on the variable, and keeps the
stream it returns as the interface's outbound message stream: the node's message towards the
variable is the variable's inbound message on this connection. `variable` is converted to an
[`AbstractVariable`](@ref) when it is not one.

The streams are lazy until the node is activated, which connects the outbound stream to the
rule's messages with [`ReactiveMP.set_stream_of_outbound_messages!`](@ref).

See also [`ReactiveMP.IndexedNodeInterface`](@ref),
[`ReactiveMP.get_stream_of_outbound_messages`](@ref),
[`ReactiveMP.get_stream_of_inbound_messages`](@ref).
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

Base.show(io::IO, interface::NodeInterface) =
    print(io, "Interface(", name(interface), ")")

israndom(interface::NodeInterface) = israndom(interface.variable)
isdata(interface::NodeInterface) = isdata(interface.variable)
isconst(interface::NodeInterface) = isconst(interface.variable)

"""
    ReactiveMP.name(interface::NodeInterface) -> Symbol
    ReactiveMP.name(interface::IndexedNodeInterface) -> Symbol
    ReactiveMP.name(localmarginal::FactorNodeLocalMarginal)

The name of an interface, `:out` or a group's name for a member of the group; the key of a
cluster's local marginal, a name or a tuple of member names. A `Symbol` is its own name.
"""
name(symbol::Symbol) = symbol
name(interface::NodeInterface) = name(interface.name)


"""
    ReactiveMP.get_stream_of_outbound_messages(interface)

The stream of the messages the node sends along `interface`, towards its variable: a
[`ReactiveMP.MessageObservable`](@ref), which is also the variable's inbound stream on this
connection.

See also [`ReactiveMP.get_stream_of_inbound_messages`](@ref).
"""
get_stream_of_outbound_messages(interface::NodeInterface) = interface.m_out

"""
    ReactiveMP.set_stream_of_outbound_messages!(interface, stream)

Connect the outbound message stream of `interface` to `stream`, the node's messages towards the
variable, which activation does.

See also [`ReactiveMP.get_stream_of_outbound_messages`](@ref).
"""
set_stream_of_outbound_messages!(interface::NodeInterface, stream) =
    connect!(get_stream_of_outbound_messages(interface), stream)

"""
    ReactiveMP.get_stream_of_inbound_messages(interface)

The stream of the messages the variable sends along `interface`, towards the node: the
variable's outbound stream on this connection, which is what the node's rules read as `m[:name]`.
For a constant it is the constant's message; for a data variable, its observations.

See also [`ReactiveMP.get_stream_of_outbound_messages`](@ref).
"""
get_stream_of_inbound_messages(interface::NodeInterface) =
    get_stream_of_outbound_messages(interface.variable, interface.message_index)

"""
    ReactiveMP.getvariable(interface) -> AbstractVariable

The variable `interface` connects to.
"""
getvariable(interface::NodeInterface) = interface.variable

"""
    ReactiveMP.IndexedNodeInterface(index::Int, interface::NodeInterface)

A member of an interface group: a [`ReactiveMP.NodeInterface`](@ref) with its index in the group,
such as member `k` of the means `m` and of the precisions `p` of a `NormalMixture`, keyed
`(:m, k)`. Its name is the group's, and its streams and variable are the wrapped interface's.

See also [`ReactiveMP.NodeInterface`](@ref).
"""
struct IndexedNodeInterface
    index::Int
    interface::NodeInterface
end

Base.show(io::IO, interface::IndexedNodeInterface) = print(
    io,
    string("IndexedInterface(", index(interface), ", ", name(interface), ")"),
)

index(interface::IndexedNodeInterface) = interface.index
name(interface::IndexedNodeInterface) = name(interface.interface)

get_stream_of_outbound_messages(interface::IndexedNodeInterface) =
    get_stream_of_outbound_messages(interface.interface)
set_stream_of_outbound_messages!(interface::IndexedNodeInterface, stream) =
    set_stream_of_outbound_messages!(interface.interface, stream)
get_stream_of_inbound_messages(interface::IndexedNodeInterface) =
    get_stream_of_inbound_messages(interface.interface)
getvariable(interface::IndexedNodeInterface) = getvariable(interface.interface)

israndom(interface::IndexedNodeInterface) = israndom(interface.interface)
isdata(interface::IndexedNodeInterface) = isdata(interface.interface)
isconst(interface::IndexedNodeInterface) = isconst(interface.interface)
