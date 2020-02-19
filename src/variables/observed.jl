export ObservedVariable, forward_message, backward_message
export update!

using Rocket

import Base: show

struct ObservedVariable <: AbstractVariable
    name   :: String
    values :: SynchronousSubject{Float64}
    left   :: Interface

    ObservedVariable(name::String, left::Interface) = begin
        variable = new(name, make_subject(Float64, mode = SYNCHRONOUS_SUBJECT_MODE), left)

        define_partner_message!(left, backward_message(variable))

        return variable
    end
end

forward_message(v::ObservedVariable)  = sum_product_message(v.left)
backward_message(v::ObservedVariable) = v.values |> map(DeterministicMessage, f -> DeterministicMessage(f))

update!(v::ObservedVariable, value::Float64) = Rocket.next!(v.values, value)

Base.show(io::IO, v::ObservedVariable) = print(io, "Observed($name, $(v.left) -> ...")
