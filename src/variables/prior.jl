export PriorVariable, forward_message, backward_message
export update!

using Rocket

import Base: show

struct PriorVariable{S} <: AbstractVariable
    name   :: String
    values :: S
    right  :: Interface
end

function PriorVariable(name::String, right::Interface)
    variable = PriorVariable(name, Subject(Float64, scheduler = Rocket.AsapScheduler()), right)

    define_partner_message!(right, forward_message(variable))

    return variable
end

forward_message(v::PriorVariable)  = v.values |> map(DeterministicMessage, f -> DeterministicMessage(f))
backward_message(v::PriorVariable) = sum_product_message(v.right)

update!(v::PriorVariable, value::Float64) = Rocket.next!(v.values, value)

Base.show(io::IO, v::PriorVariable) = print(io, "Prior($name, ... -> $(v.right)")
