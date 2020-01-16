using Rx

export AbstractVariable, inference
export RandomVariable, ConstantVariable, ObservedVariable, EstimatedVariable
export update!, forward_message, backward_message
export random_variable, constant_variable, observed_variable, estimated_variable

abstract type AbstractVariable{F, B} end

function inference(variable::V) where { V <: AbstractVariable{F, B} } where { F <: AbstractMessage } where { B <: AbstractMessage }
    return combineLatest(forward_message(variable), backward_message(variable)) |> map(AbstractMessage, (t) -> multiply(t[1], t[2]))
end

struct RandomVariable{F, B} <: AbstractVariable{F, B}
    name :: String

    left  :: InterfaceOut{F, B}
    right :: InterfaceIn{B, F}

    RandomVariable{F, B}(name::String, left::InterfaceOut{F, B}, right::InterfaceIn{B, F}) where { F <: AbstractMessage } where { B <: AbstractMessage } = begin
        variable = new(name, left, right)

        define_joint!(left, backward_message(variable))
        define_joint!(right, forward_message(variable))

        return variable
    end
end

random_variable(name::String, left::InterfaceOut{F, B}, right::InterfaceIn{B, F}) where { F <: AbstractMessage } where { B <: AbstractMessage } = RandomVariable{F, B}(name, left, right)

forward_message(v::RandomVariable{F, B}) where F where B  = sum_product(v.left)
backward_message(v::RandomVariable{F, B}) where F where B = sum_product(v.right)

struct ConstantVariable{B} <: AbstractVariable{DeterministicMessage, B}
    name  :: String
    value :: Float64

    right :: InterfaceIn{B, DeterministicMessage}

    ConstantVariable{B}(name::String, value::Float64, right::InterfaceIn{B, DeterministicMessage}) where { B <: AbstractMessage } = begin
        variable = new(name, value, right)

        define_joint!(right, forward_message(variable))

        return variable
    end
end

constant_variable(name::String, value::Float64, right::InterfaceIn{B, DeterministicMessage}) where { B <: AbstractMessage } = ConstantVariable{B}(name, value, right)

forward_message(v::ConstantVariable{B})  where B = of(DeterministicMessage(v.value))
backward_message(v::ConstantVariable{B}) where B = sum_product(v.right_interface)

struct ObservedVariable{F} <: AbstractVariable{F, DeterministicMessage}
    name   :: String
    values :: SyncSubject{Float64}

    left   :: InterfaceOut{F, DeterministicMessage}

    ObservedVariable{F}(name::String, left::InterfaceOut{F, DeterministicMessage}) where { F <: AbstractMessage } = begin
        variable = new(name, SyncSubject{Float64}(), left)

        define_joint!(left, backward_message(variable))

        return variable
    end
end

observed_variable(name::String, left::InterfaceOut{F, DeterministicMessage}) where { F <: AbstractMessage } = ObservedVariable{F}(name, left)

@CreateMapOperator(ObservedBackward, Float64, DeterministicMessage, (f::Float64) -> DeterministicMessage(f))

forward_message(v::ObservedVariable{F})  where F = sum_product(v.left_interface)
backward_message(v::ObservedVariable{F}) where F = v.values |> ObservedBackwardMapOperator()

update!(variable::ObservedVariable, value::Float64) = next!(variable.values, value)

struct EstimatedVariable{B} <: AbstractVariable{DeterministicMessage, B}
    name   :: String
    values :: SyncSubject{Float64}

    right  :: InterfaceIn{B, DeterministicMessage}

    EstimatedVariable{B}(name::String, right::InterfaceIn{B, DeterministicMessage}) where { B <: AbstractMessage } = begin
        variable = new(name, SyncSubject{Float64}(), right)

        define_joint!(right, forward_message(variable))

        return variable
    end
end

estimated_variable(name::String, right::InterfaceIn{B, DeterministicMessage}) where { B <: AbstractMessage } = EstimatedVariable{B}(name, right)

@CreateMapOperator(EstimatedForward, Float64, DeterministicMessage, (f::Float64) -> DeterministicMessage(f))

forward_message(v::EstimatedVariable{B})  where B = v.values |> EstimatedForwardMapOperator()
backward_message(v::EstimatedVariable{B}) where B = sum_product(v.right_interface)

update!(variable::EstimatedVariable, value::Float64) = next!(variable.values, value)
