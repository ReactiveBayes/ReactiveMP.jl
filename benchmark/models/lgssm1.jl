# Simple Linear Gaussian State Space Model

module LGSSM1Benchmark

using Rocket
using ReactiveMP
using GraphPPL
using Distributions

@model function lgssm(n)
    x = randomvar(n)
    y = datavar(Float64, n)
    c = constvar(1.0)
    P = constvar(1.0)

    x_prior ~ NormalMeanVariance(0.0, 100.0) 
    x_prev = x_prior
    
    for i in 1:n
        x[i] ~ x_prev + c
        y[i] ~ NormalMeanVariance(x[i], P)
        x_prev = x[i]
    end

    return x, y
end

generate_input(rng, n) = collect(1:n) + rand(rng, Normal(0.0, sqrt(1.0)), n);

function benchmark(input)
    n = length(input)
    
    _, (x, y) = lgssm(n);

    x_buffer  = buffer(Marginal, n)
    marginals = getmarginals(x)
    
    subscription = subscribe!(marginals, x_buffer)
    
    update!(y, input)
    
    unsubscribe!(subscription)
    
    return getvalues(x_buffer)
end

end