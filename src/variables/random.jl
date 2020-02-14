export RandomVariable, forward_message, backward_message

using Rocket

import Base: show

struct RandomVariable <: AbstractVariable
    name  :: String
    left  :: Interface
    right :: Interface

    RandomVariable(name::String, left::Interface, right::Interface) = begin
        variable = new(name, left, right)

        define_partner_message!(left, backward_message(variable))
        define_partner_message!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::RandomVariable)  = sum_product_message(v.left)
backward_message(v::RandomVariable) = sum_product_message(v.right)

Base.show(io::IO, v::RandomVariable) = print(io, "Random($(v.name), $(v.left) -> $(v.right)")
