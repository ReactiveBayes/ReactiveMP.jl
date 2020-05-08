export ObservedVariable, forward_message, backward_message
export update!

using Rocket

import Base: show

struct ObservedVariable{S} <: AbstractVariable
    name   :: String
    values :: S
    left   :: Interface
end

function ObservedVariable(name::String, left::Interface)
    variable = ObservedVariable(name, Subject(Float64, scheduler = Rocket.AsapScheduler()), left)

    define_partner_message!(left, backward_message(variable))

    return variable
end

forward_message(v::ObservedVariable)  = sum_product_message(v.left)
backward_message(v::ObservedVariable) = v.values |> map(DeterministicMessage, f -> DeterministicMessage(f))

update!(v::ObservedVariable, value::Float64) = Rocket.next!(v.values, value)

Base.show(io::IO, v::ObservedVariable) = print(io, "Observed($name, $(v.left) -> ...")
