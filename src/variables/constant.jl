export ConstantVariable, forward_message, backward_message

using Rocket

import Base: show

struct ConstantVariable <: AbstractVariable
    name  :: String
    value :: Float64
    right :: Interface

    ConstantVariable(name::String, value::Float64, right::Interface) = begin
        variable = new(name, value, right)

        define_partner_message!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::ConstantVariable)  = of(DeterministicMessage(v.value))
backward_message(v::ConstantVariable) = sum_product_message(v.right_interface)

Base.show(io::IO, v::ConstantVariable) = print(io, "Constant($(v.name), Const($(v.value)) -> $(v.right)")
