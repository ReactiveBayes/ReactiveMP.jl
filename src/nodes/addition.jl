export AdditionNode

using Rocket

additionOutForward(l ,r)  = combineLatest((l, r), true, (AbstractMessage, calculate_addition_out))
additionIn1Backward(l, r) = combineLatest((l, r), true, (AbstractMessage, calculate_addition_in1))
additionIn2Backward(l, r) = combineLatest((l, r), true, (AbstractMessage, calculate_addition_in2))

struct AdditionNode <: AbstractDeterministicNode
    name :: String
    in1  :: Interface
    in2  :: Interface
    out  :: Interface

    AdditionNode(name::String) = begin
        in1 = Interface("[$name]: in1")
        in2 = Interface("[$name]: in2")
        out = Interface("[$name]: out")

        # Forward message over the out
        define_sum_product_message!(out, additionOutForward(partner_message(in1), partner_message(in2)) |> share(mode = SYNCHRONOUS_SUBJECT_MODE))
        # define_sum_product_message!(out, additionOutForward(partner_message(in1), partner_message(in2)))

        # Backward message over the in1
        define_sum_product_message!(in1, additionIn1Backward(partner_message(out), partner_message(in2)))

        # Backward message over the in2
        define_sum_product_message!(in2, additionIn2Backward(partner_message(out), partner_message(in1)))

        return new(name, in1, in2, out)
    end
end

### Out ###

function calculate_addition_out(t::Tuple)
    return calculate_addition_out(t[1], t[2])
end

function calculate_addition_out(m1::DeterministicMessage, m2::DeterministicMessage)
    return DeterministicMessage(m1.value + v2.value)
end

function calculate_addition_out(m1::StochasticMessage{D}, m2::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(m1.distribution) + mean(m2.distribution), sqrt(var(m1.distribution) + var(m2.distribution))))
end

function calculate_addition_out(m1::StochasticMessage{D}, m2::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(m1.distribution) + m2.value, std(m1.distribution)))
end

function calculate_addition_out(m1::DeterministicMessage, m2::StochasticMessage{D}) where { D <: Normal{Float64} }
    return calculate_addition_out(m2, m1)
end

### In 1 ###

function calculate_addition_in1(t::Tuple)
    return calculate_addition_in1(t[1], t[2])
end

function calculate_addition_in1(out_m::DeterministicMessage, in2_m::DeterministicMessage)
    return DeterministicMessage(out_m.value - in2_m.value)
end

function calculate_addition_in1(out_m::StochasticMessage{D}, in2_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - mean(in2_m.distribution), sqrt(var(out_m.distribution) + var(in2_m.distribution))))
end

function calculate_addition_in1(out_m::StochasticMessage{D}, in2_m::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - in2_m.value, std(out_m.distribution)))
end

function calculate_addition_in1(out_m::DeterministicMessage, in2_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(out_m.value - mean(in2_m.distribution), std(in2_m.distribution)))
end

### In 2 ###

function calculate_addition_in2(t::Tuple)
    return calculate_addition_in2(t[1], t[2])
end

function calculate_addition_in2(out_m::DeterministicMessage, in1_m::DeterministicMessage)
    return DeterministicMessage(out_m.value - in1_m.value)
end

function calculate_addition_in2(out_m::StochasticMessage{D}, in1_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - mean(in1_m.distribution), sqrt(var(out_m.distribution) + var(in1_m.distribution))))
end

function calculate_addition_in2(out_m::StochasticMessage{D}, in1_m::DeterministicMessage) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(mean(out_m.distribution) - in1_m.value, std(out_m.distribution)))
end

function calculate_addition_in2(out_m::DeterministicMessage, in1_m::StochasticMessage{D}) where { D <: Normal{Float64} }
    return StochasticMessage(Normal(out_m.value - mean(in1_m.distribution), std(in1_m.distribution)))
end
