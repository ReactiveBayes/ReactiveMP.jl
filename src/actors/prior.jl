

using Rocket

# Broken functionaliy 

# struct PriorActor{D, P} <: Actor{D}
#     name  :: Symbol
#     prior :: P
# end

# prioractor(::Type{D}, name::Symbol, prior::P) where { D, P } = PriorActor{D, P}(name, prior)

# function Rocket.on_next!(actor::PriorActor{D}, data::D) where D
#     update!(actor.prior, as_message(data))
# end

# function Rocket.on_error!(actor::PriorActor, err)
#     println(stderr, "Error in prior actor $(actor.name): $err")
# end

# function Rocket.on_complete!(::PriorActor)
#     return nothing
# end

# link(datavar::DataVariable, source::S) where S = check_link(datavar, as_subscribable(S), source)

# check_link(datavar::DataVariable, ::InvalidSubscribableTrait, source)              = throw(InvalidSubscribableTraitUsageError(source))
# check_link(datavar::DataVariable, ::SimpleSubscribableTrait{D}, source)    where D = _link(datavar, D, source)
# check_link(datavar::DataVariable, ::ScheduledSubscribableTrait{D}, source) where D = _link(datavar, D, source)

# _link(datavar::DataVariable, ::Type{D}, source) where D = subscribe!(source |> discontinue(), prioractor(D, name(datavar), datavar))
