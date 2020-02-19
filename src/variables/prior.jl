export PriorVariable, forward_message, backward_message
export update!

using Rocket

import Base: show

struct PriorVariable <: AbstractVariable
    name   :: String
    values :: SynchronousSubject{Float64}
    right  :: Interface

    PriorVariable(name::String, right::Interface) = begin
        variable = new(name, make_subject(Float64, mode = SYNCHRONOUS_SUBJECT_MODE), right)

        define_partner_message!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::PriorVariable)  = v.values |> map(DeterministicMessage, f -> DeterministicMessage(f))
backward_message(v::PriorVariable) = sum_product_message(v.right)

update!(v::PriorVariable, value::Float64) = Rocket.next!(v.values, value)

Base.show(io::IO, v::PriorVariable) = print(io, "Prior($name, ... -> $(v.right)")
