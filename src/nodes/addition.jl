using Rx

export AdditionNode, addition_node

struct AdditionOutForwardMapOperator{In1J, In2J, OutS} <: TypedOperator{Tuple{In1J, In2J}, OutS} end

function Rx.on_call!(::Type{Tuple{In1J, In2J}}, ::Type{OutS}, operator::AdditionOutForwardMapOperator{In1J, In2J, OutS}, source) where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage }
    return ProxyObservable{OutS}(source, AdditionOutForwardMapOperatorProxy{In1J, In2J, OutS}())
end

function Rx.on_call!(::Type{Tuple{In1J, In2J}}, ::Type{OutS}, operator::AdditionOutForwardMapOperator{In1J, In2J, OutS}, source::SingleObservable{Tuple{In1J, In2J}}) where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage }
    return SingleObservable{OutS}(calculate_addition_out(source.value[1], source.value[2]))
end

struct AdditionOutForwardMapOperatorProxy{In1J, In2J, OutS} <: ActorProxy end

function Rx.actor_proxy!(proxy::AdditionOutForwardMapOperatorProxy{In1J, In2J, OutS}, actor::A) where { A <: AbstractActor{OutS} } where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage }
    return AdditionOutForwardMapOperatorActor{In1J, In2J, OutS, A}(actor)
end

struct AdditionOutForwardMapOperatorActor{ In1J, In2J, OutS, A <: AbstractActor{OutS} } <: Actor{Tuple{In1J, In2J}}
    actor :: A
end

function Rx.on_next!(actor::AdditionOutForwardMapOperatorActor{In1J, In2J, OutS, A}, data::Tuple{In1J, In2J}) where { A <: AbstractActor{OutS} } where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage }
    next!(actor.actor, calculate_addition_out(data[1], data[2]))
end

function Rx.on_error!(actor::AdditionOutForwardMapOperatorActor{In1J, In2J, OutS, A}, err) where { A <: AbstractActor{OutS} } where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage }
    error!(actor.actor, err)
end

function Rx.on_complete!(actor::AdditionOutForwardMapOperatorActor{In1J, In2J, OutS, A}) where { A <: AbstractActor{OutS} } where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage }
    complete!(actor.actor)
end

## ------------------ ##

struct AdditionIn1BackwardMapOperator{OutJ, In2J, In1S} <: TypedOperator{Tuple{OutJ, In2J}, In1S} end

function Rx.on_call!(::Type{Tuple{OutJ, In2J}}, ::Type{In1S}, operator::AdditionIn1BackwardMapOperator{OutJ, In2J, In1S}, source) where { OutJ <: AbstractMessage } where { In2J <: AbstractMessage } where { In1S <: AbstractMessage }
    return ProxyObservable{In1S}(source, AdditionIn1BackwardMapOperatorProxy{OutJ, In2J, In1S}())
end

function Rx.on_call!(::Type{Tuple{OutJ, In2J}}, ::Type{In1S}, operator::AdditionIn1BackwardMapOperator{OutJ, In2J, In1S}, source::SingleObservable{Tuple{OutJ, In2J}}) where { OutJ <: AbstractMessage } where { In2J <: AbstractMessage } where { In1S <: AbstractMessage }
    return SingleObservable{In1S}(calculate_addition_in1(source.value[1], source.value[2]))
end

struct AdditionIn1BackwardMapOperatorProxy{OutJ, In2J, In1S} <: ActorProxy end

function Rx.actor_proxy!(proxy::AdditionIn1BackwardMapOperatorProxy{OutJ, In2J, In1S}, actor::A) where { A <: AbstractActor{In1S} } where { OutJ <: AbstractMessage } where { In2J <: AbstractMessage } where { In1S <: AbstractMessage }
    return AdditionIn1BackwardMapOperatorActor{OutJ, In2J, In1S, A}(actor)
end

struct AdditionIn1BackwardMapOperatorActor{ OutJ, In2J, In1S, A <: AbstractActor{In1S} } <: Actor{Tuple{OutJ, In2J}}
    actor :: A
end

function Rx.on_next!(actor::AdditionIn1BackwardMapOperatorActor{OutJ, In2J, In1S, A}, data::Tuple{OutJ, In2J}) where { A <: AbstractActor{In1S} } where { OutJ <: AbstractMessage } where { In2J <: AbstractMessage } where { In1S <: AbstractMessage }
    next!(actor.actor, calculate_addition_in1(data[1], data[2]))
end

function Rx.on_error!(actor::AdditionIn1BackwardMapOperatorActor{OutJ, In2J, In1S, A}, err) where { A <: AbstractActor{In1S} } where { OutJ <: AbstractMessage } where { In2J <: AbstractMessage } where { In1S <: AbstractMessage }
    error!(actor.actor, err)
end

function Rx.on_complete!(actor::AdditionIn1BackwardMapOperatorActor{OutJ, In2J, In1S, A}) where { A <: AbstractActor{In1S} } where { OutJ <: AbstractMessage } where { In2J <: AbstractMessage } where { In1S <: AbstractMessage }
    complete!(actor.actor)
end


struct AdditionNode{In1S, In1J, In2S, In2J, OutS, OutJ} <: AbstractFactorNode
    name :: String

    in1 :: InterfaceIn{In1S, In1J}
    in2 :: InterfaceIn{In2S, In2J}
    out :: InterfaceOut{OutS, OutJ}

    AdditionNode{In1S, In1J, In2S, In2J, OutS, OutJ}(name::String) where { In1S <: AbstractMessage } where { In1J <: AbstractMessage } where { In2S <: AbstractMessage } where { In2J <: AbstractMessage } where { OutS <: AbstractMessage } where { OutJ <: AbstractMessage }  = begin
        in1 = InterfaceIn{In1S, In1J}("[$name] in1InterfaceIn")
        in2 = InterfaceIn{In2S, In2J}("[$name] in2InterfaceIn")
        out = InterfaceOut{OutS, OutJ}("[$name] outInterfaceOut")

        # Forward message over the out
        define_sum_product!(out, combineLatest(in1.joint_message, in2.joint_message) |> AdditionOutForwardMapOperator{In1J, In2J, OutS}())

        # Backward message over the in1
        define_sum_product!(in1, combineLatest(out.joint_message, in2.joint_message) |> AdditionIn1BackwardMapOperator{OutJ, In2J, In1S}())

        # Backward message over the in2
        define_sum_product!(in2, combineLatest(out.joint_message, in1.joint_message) |> map(In2S, (d::Tuple{OutJ, In1J}) -> calculate_addition_in2(d[1], d[2])))

        return new(name, in1, in2, out)
    end
end

function addition_node(name::String, ::Type{In1J}, ::Type{In2J}, ::Type{OutJ}) where { In1J <: AbstractMessage } where { In2J <: AbstractMessage } where { OutJ <: AbstractMessage }
    return addition_node(name, In1J, addition_type_in1(OutJ, In2J), In2J, addition_type_in2(OutJ, In1J), OutJ, addition_type_out(In1J, In2J))
end

function addition_node(name::String, ::Type{In1J}, ::Type{In1S}, ::Type{In2J}, ::Type{In2S}, ::Type{OutJ}, ::Type{OutS}) where { In1J <: AbstractMessage } where { In1S <: AbstractMessage } where { In2J <: AbstractMessage } where { In2S <: AbstractMessage } where { OutJ <: AbstractMessage } where { OutS <: AbstractMessage }
    return AdditionNode{In1S, In1J, In2S, In2J, OutS, OutJ}(name)
end

### Out ###

addition_type_out(::Type{In1}, ::Type{In2}) where { In1 <: DeterministicMessage } where { In2 <: DeterministicMessage } = DeterministicMessage
addition_type_out(::Type{In1}, ::Type{In2}) where { In1 <: StochasticMessage{D} } where { In2 <: StochasticMessage{D} } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}
addition_type_out(::Type{In1}, ::Type{In2}) where { In1 <: DeterministicMessage } where { In2 <: StochasticMessage{D} } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}
addition_type_out(::Type{In1}, ::Type{In2}) where { In1 <: StochasticMessage{D} } where { In2 <: DeterministicMessage } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}

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

addition_type_in1(::Type{Out}, ::Type{In2}) where { Out <: DeterministicMessage } where { In2 <: DeterministicMessage } = DeterministicMessage
addition_type_in1(::Type{Out}, ::Type{In2}) where { Out <: StochasticMessage{D} } where { In2 <: StochasticMessage{D} } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}
addition_type_in1(::Type{Out}, ::Type{In2}) where { Out <: StochasticMessage{D} } where { In2 <: DeterministicMessage } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}
addition_type_in1(::Type{Out}, ::Type{In2}) where { Out <: DeterministicMessage } where { In2 <: StochasticMessage{D} } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}

calculate_addition_in1(out_m::DeterministicMessage, in2_m::DeterministicMessage) = DeterministicMessage(out_m.value - in2_m.value)

function calculate_addition_in1(out_m::StochasticMessage{D}, in2_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    # TODO ??
    return StochasticMessage(Normal(mean(out_m.distribution) - mean(in2_m.distribution), sqrt(var(out_m.distribution) + var(in2_m.distribution))))
end

function calculate_addition_in1(out_m::StochasticMessage{D}, in2_m::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - in2_m.value, std(out_m.distribution)))
end

function calculate_addition_in1(out_m::DeterministicMessage, in2_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(out_m.value - mean(in2_m.distribution), std(in2_m.distribution)))
end

### In 2 ###

addition_type_in2(::Type{Out}, ::Type{In1}) where { Out <: DeterministicMessage } where { In1 <: DeterministicMessage } = DeterministicMessage
addition_type_in2(::Type{Out}, ::Type{In1}) where { Out <: StochasticMessage{D} } where { In1 <: StochasticMessage{D} } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}
addition_type_in2(::Type{Out}, ::Type{In1}) where { Out <: StochasticMessage{D} } where { In1 <: DeterministicMessage } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}
addition_type_in2(::Type{Out}, ::Type{In1}) where { Out <: DeterministicMessage } where { In1 <: StochasticMessage{D} } where { D <: Normal{Float64} } = StochasticMessage{Normal{Float64}}

calculate_addition_in2(out_m::DeterministicMessage, in1_m::DeterministicMessage) = DeterministicMessage(out_m.value - in1_m.value)

function calculate_addition_in2(out_m::StochasticMessage{D}, in1_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    # TODO ??
    return StochasticMessage(Normal(mean(out_m.distribution) - mean(in1_m.distribution), sqrt(var(out_m.distribution) + var(in1_m.distribution))))
end

function calculate_addition_in2(out_m::StochasticMessage{D}, in1_m::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - in1_m.value, std(out_m.distribution)))
end

function calculate_addition_in2(out_m::DeterministicMessage, in1_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(out_m.value - mean(in1_m.distribution), std(in1_m.distribution)))
end
