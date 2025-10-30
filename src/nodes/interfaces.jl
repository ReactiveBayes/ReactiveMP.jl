"""
    NodeInterface

`NodeInterface` object represents a single node-variable connection.
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

    function NodeInterface(name::Symbol, variable::Any)
        return NodeInterface(name, convert(AbstractVariable, variable))
    end
end

Base.show(io::IO, interface::NodeInterface) = print(io, "Interface(", name(interface), ")")

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
    messageout(interface)

Returns an outbound messages stream from the given interface.
"""
messageout(interface::NodeInterface) = interface.m_out

"""
    messagein(interface)

Returns an inbound messages stream from the given interface.
"""
messagein(interface::NodeInterface) = messageout(interface.variable, interface.message_index)

"""
    getvariable(interface)

Returns a variable connected to the given interface.
"""
getvariable(interface::NodeInterface) = interface.variable

"""
    IndexedNodeInterface

`IndexedNodeInterface` object represents a repetative node-variable connection.
Used in cases when a node may connect to a different number of random variables with the same name, e.g. means and precisions of a Gaussian Mixture node.
"""
struct IndexedNodeInterface
    index     :: Int
    interface :: NodeInterface
end

Base.show(io::IO, interface::IndexedNodeInterface) = print(io, string("IndexedInterface(", index(interface), ", ", name(interface), ")"))

index(interface::IndexedNodeInterface) = interface.index
name(interface::IndexedNodeInterface)  = name(interface.interface)
tag(interface::IndexedNodeInterface)   = (tag(interface.interface), index(interface))

messageout(interface::IndexedNodeInterface) = messageout(interface.interface)
messagein(interface::IndexedNodeInterface) = messagein(interface.interface)
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

Base.show(io::IO, manyof::ManyOf) = print(io, "ManyOf(", join(manyof.collection, ",", ""), ")")

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
        return string("ManyOf{", N, ", ", rule_method_error_type_nameof(elt), "}")
    end

    # Otherwise produce the Union of element type names:
    unions = join(map(r -> rule_method_error_type_nameof(dropproxytype(r)), fts), ",")
    return string("ManyOf{", N, ", Union{", unions, "}}")
end

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
