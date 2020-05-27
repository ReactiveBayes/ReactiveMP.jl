export prioractor, link

using Rocket

struct PriorActor{D, P} <: Actor{D}
    name  :: Symbol
    prior :: P
end

prioractor(::Type{D}, name::Symbol, prior::P) where { D, P } = PriorActor{D, P}(name, prior)

function Rocket.on_next!(actor::PriorActor{D}, data::D) where D
    update!(actor.prior, getdata(data))
end

function Rocket.on_error!(actor::PriorActor, err)
    println(stderr, "Error in prior actor $(actor.name): $err")
end

function Rocket.on_complete!(::PriorActor)
    return nothing
end

link(datavar::DataVariable, source::S) where S = link(datavar, as_subscribable(S), source)

link(datavar::DataVariable, ::InvalidSubscribable, source)               = throw(InvalidSubscribableTraitUsageError(source))
link(datavar::DataVariable, ::ValidSubscribableTrait{D}, source) where D = subscribe!(source |> discontinue(), prioractor(D, name(datavar), datavar))
