module KalmanFilter

using Distributions
using Rocket
using ReactiveMP
using BenchmarkTools

import Base: show

mutable struct InferenceActor <: Actor{Message}
    index       :: Int
    size        :: Int
    data        :: Vector{Float64}
    messages    :: Vector{Message}
    communicate :: Channel{Tuple{Float64, Float64}}

    y          :: ObservedVariable
    e_mean     :: PriorVariable
    e_variance :: PriorVariable

    InferenceActor(data::Vector{Float64}, y::ObservedVariable, e_mean::PriorVariable, e_variance::PriorVariable) = begin
        size  = length(data)
        messages = Vector{Message}(undef, size)

        actor = new(1, size, data, messages, Channel{Tuple{Float64, Float64}}(Inf), y, e_mean, e_variance)

        task = @async begin
            while true
                u = take!(actor.communicate)
                update!(actor, u[1], u[2])
            end
        end

        bind(actor.communicate, task)

        return actor
    end
end

function update!(actor::InferenceActor, mean::Float64, variance::Float64)
    next!(actor.y.values, actor.data[actor.index])
    next!(actor.e_mean.values, mean)
    next!(actor.e_variance.values, variance)
end

function stop!(actor::InferenceActor)
    complete!(actor.y.values)
    complete!(actor.e_mean.values)
    complete!(actor.e_variance.values)
end

function Rocket.on_next!(actor::InferenceActor, data::Message)
    m = mean(data.distribution)
    v = var(data.distribution)

    actor.messages[actor.index] = data

    actor.index += 1

    if actor.index < actor.size + 1
        put!(actor.communicate, (m, v))
    else
        stop!(actor)
    end
end

Rocket.on_error!(actor::InferenceActor, err) = error(err)
Rocket.on_complete!(actor::InferenceActor)   = close(actor.communicate)

Base.show(io::IO, actor::InferenceActor) = print(io, "InferenceActor")

function kalman()
    N = 5000
    data = collect(1:N) + sqrt(200.0) * randn(N);

    x_prev_add   = AdditionNode("x_prev_add");
    add_1        = ConstantVariable("add_1", 1.0, x_prev_add.in2);

    x_prev_prior = GaussianMeanVarianceNode("x_prev_prior");
    x_prev_m     = PriorVariable("x_prev_m", x_prev_prior.mean);
    x_prev_v     = PriorVariable("x_prev_v", x_prev_prior.variance);

    x_prev = RandomVariable("x_prev", x_prev_prior.value, x_prev_add.in1);

    noise_node     = GaussianMeanVarianceNode("noise_node");
    noise_mean     = ConstantVariable("noise_mean", 0.0, noise_node.mean);
    noise_variance = ConstantVariable("noise_variance", 200.0, noise_node.variance);

    add_x_and_noise = AdditionNode("add_x_and_noise");

    x = RandomVariable("x", x_prev_add.out, add_x_and_noise.in1);
    n = RandomVariable("n", noise_node.value, add_x_and_noise.in2);
    y = ObservedVariable("y", add_x_and_noise.out);

    actor  = InferenceActor(data, y, x_prev_m, x_prev_v);
    synced = sync(actor)

    @async begin
        try
            subscribe!(inference(x), synced)
            update!(actor, 0.0, 1000.0)
        catch e
            println(e)
        end
    end

    wait(synced)

    return actor.messages
end

function julia_main()::Cint
    @time kalman();
    @time kalman();
    @btime kalman();

    values = kalman()

    println(values[1:5])
    println(values[end-5:end])
    return 0
end

end # module
