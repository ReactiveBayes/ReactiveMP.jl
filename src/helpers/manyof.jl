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