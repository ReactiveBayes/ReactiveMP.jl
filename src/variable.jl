using Rx

export AbstractVariable, inference
export RandomVariable, ConstantVariable, ObservedVariable, EstimatedVariable

abstract type AbstractVariable end

function inference(variable::AbstractVariable)
    return combineLatest(forward_message(variable), backward_message(variable)) |> map(AbstractMessage, (t) -> multiply(t[1], t[2]))
end

struct RandomVariable <: AbstractVariable
    name :: String

    left_interface  :: InterfaceOut
    right_interface :: InterfaceIn

    RandomVariable(name, left, right) = begin
        variable = new(name::String, left::InterfaceOut, right::InterfaceIn)

        define_joint!(left, backward_message(variable))
        define_joint!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::RandomVariable)  = sum_product(v.left_interface)
backward_message(v::RandomVariable) = sum_product(v.right_interface)

struct ConstantVariable <: AbstractVariable
    name  :: String
    value :: Float64

    right_interface :: InterfaceIn

    ConstantVariable(name::String, value::Float64, right::InterfaceIn) = begin
        variable = new(name, value, right)

        define_joint!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::ConstantVariable)  = SingleObservable{AbstractMessage}(DeterministicMessage(v.value))
backward_message(v::ConstantVariable) = sum_product(v.right_interface)

struct ObservedVariable <: AbstractVariable
    name   :: String
    values :: Subject{Float64}

    left_interface :: InterfaceOut

    ObservedVariable(name::String, left::InterfaceOut) = begin
        variable = new(name, Subject{Float64}(), left)

        define_joint!(left, backward_message(variable))

        return variable
    end
end

forward_message(v::ObservedVariable)  = sum_product(v.left_interface)
backward_message(v::ObservedVariable) = v.values |> map(AbstractMessage, (f) -> DeterministicMessage(f))

next!(variable::ObservedVariable, value::Float64) = next!(variable.values, value)

struct EstimatedVariable <: AbstractVariable
    name   :: String
    values :: Subject{Float64}

    right_interface :: InterfaceIn

    EstimatedVariable(name::String, right::InterfaceIn) = begin
        variable = new(name, Subject{Float64}(), right)

        define_joint!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::EstimatedVariable)  = v.values |> map(AbstractMessage, (f) -> DeterministicMessage(f))
backward_message(v::EstimatedVariable) = sum_product(v.right_interface)

next!(variable::EstimatedVariable, value::Float64) = next!(variable.values, value)
