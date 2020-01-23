using Rx

export AbstractVariable, inference
export RandomVariable, ConstantVariable, ObservedVariable, EstimatedVariable
export update!, forward_message, backward_message
export random_variable, constant_variable, observed_variable, estimated_variable

abstract type AbstractVariable end

@CreateMapOperator(Inference, Tuple{AbstractMessage, AbstractMessage}, AbstractMessage, (t) -> multiply(t[1], t[2]))

function inference(variable)
    return combineLatest(forward_message(variable), backward_message(variable)) |> InferenceMapOperator()
end

struct RandomVariable <: AbstractVariable
    name  :: String
    left  :: InterfaceOut
    right :: InterfaceIn

    RandomVariable(name::String, left::InterfaceOut, right::InterfaceIn) = begin
        variable = new(name, left, right)

        define_joint!(left, backward_message(variable))
        define_joint!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::RandomVariable)  = sum_product(v.left)
backward_message(v::RandomVariable) = sum_product(v.right)

struct ConstantVariable <: AbstractVariable
    name  :: String
    value :: Float64
    right :: InterfaceIn

    ConstantVariable(name::String, value::Float64, right::InterfaceIn) = begin
        variable = new(name, value, right)

        define_joint!(right, forward_message(variable))

        return variable
    end
end

forward_message(v::ConstantVariable)  = of(DeterministicMessage(v.value))
backward_message(v::ConstantVariable) = sum_product(v.right_interface)

struct ObservedVariable <: AbstractVariable
    name   :: String
    values :: SynchronousSubject{Float64}
    left   :: InterfaceOut

    ObservedVariable(name::String, left::InterfaceOut) = begin
        variable = new(name, SynchronousSubject{Float64}(), left)

        define_joint!(left, backward_message(variable))

        return variable
    end
end

@CreateMapOperator(ObservedBackward, Float64, DeterministicMessage, (f::Float64) -> DeterministicMessage(f))

forward_message(v::ObservedVariable)  = sum_product(v.left_interface)
backward_message(v::ObservedVariable) = v.values |> ObservedBackwardMapOperator()

update!(variable::ObservedVariable, value::Float64) = next!(variable.values, value)

struct EstimatedVariable <: AbstractVariable
    name   :: String
    values :: SynchronousSubject{Float64}
    right  :: InterfaceIn

    EstimatedVariable(name::String, right::InterfaceIn) = begin
        variable = new(name, SynchronousSubject{Float64}(), right)

        define_joint!(right, forward_message(variable))

        return variable
    end
end

@CreateMapOperator(EstimatedForward, Float64, DeterministicMessage, (f::Float64) -> DeterministicMessage(f))

forward_message(v::EstimatedVariable)  = v.values |> EstimatedForwardMapOperator()
backward_message(v::EstimatedVariable) = sum_product(v.right_interface)

update!(variable::EstimatedVariable, value::Float64) = next!(variable.values, value)
