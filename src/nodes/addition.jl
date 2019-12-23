using Rx

export AdditionNode

@CreateMapOperator(ForwardOut, Tuple{AbstractMessage, AbstractMessage}, AbstractMessage, (d) -> calculate_addition_out(d[1], d[2]))
@CreateMapOperator(BackwardIn1, Tuple{AbstractMessage, AbstractMessage}, AbstractMessage, (d) -> calculate_addition_in1(d[1], d[2]))

struct AdditionNode <: AbstractFactorNode
    name :: String

    in1 :: InterfaceIn
    in2 :: InterfaceIn
    out :: InterfaceOut

    AdditionNode(name::String) = begin
        in1 = InterfaceIn("[$name] in1InterfaceIn")
        in2 = InterfaceIn("[$name] in2InterfaceIn")
        out = InterfaceOut("[$name] outInterfaceOut")

        # Forward message over the out
        define_sum_product!(out, combineLatest(in1.joint_message, in2.joint_message) |> ForwardOutMapOperator())

        # Backward message over the in1
        define_sum_product!(in1, combineLatest(out.joint_message, in2.joint_message) |> BackwardIn1MapOperator())

        # Backward message over the in2
        define_sum_product!(in2, throwError("[$name]: Sum product message over the in2 in addition node is not implemented at all because I am too lazy", AbstractMessage))

        return new(name, in1, in2, out)
    end
end

### Out ###

calculate_addition_out(m1::DeterministicMessage, m2::DeterministicMessage) = DeterministicMessage(m1.value + v2.value)

function calculate_addition_out(m1::StochasticMessage{D}, m2::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(m1.distribution) + mean(m2.distribution), sqrt(var(m1.distribution) + var(m2.distribution))))
end

function calculate_addition_out(m1::StochasticMessage{D}, m2::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(m1.distribution) + m2.value, std(m1.distribution)))
end

function calculate_addition_out(m1::DeterministicMessage, m2::StochasticMessage{D}) where { D <: Normal{Float64} }
    calculate_addition_out(m2, m1)
end

### In 1 ###

calculate_addition_in1(out_m::DeterministicMessage, in2_m::DeterministicMessage) = DeterministicMessage(out_m.value - in2_m.value)

function calculate_addition_in1(out_m::StochasticMessage{D}, in2_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - mean(in2_m.distribution), sqrt(var(out_m.distribution) - var(in2_m.distribution))))
end

function calculate_addition_in1(out_m::StochasticMessage{D}, in2_m::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - in2_m.value, std(out_m.distribution)))
end

function calculate_addition_in1(out_m::DeterministicMessage, in2_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(out_m.value - mean(in2_m.distribution), std(in2_m.distribution)))
end
