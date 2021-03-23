# Multivariate Linear Gaussian State Space Model

module LGSSM2Benchmark

using Rocket
using ReactiveMP
using GraphPPL
using Distributions

@model function lgssm(n)
    θ = π / 35

    x = randomvar(n)
    y = datavar(Vector{Float64}, n)
    
    x_prior ~ MvNormalMeanCovariance(zeros(2), Matrix(Diagonal(100.0 * ones(2))))
    x_prev = x_prior
    
    Q = constvar(Matrix(Diagonal(1.0 * ones(2))))
    P = constvar(Matrix(Diagonal(1.0 * ones(2))))
    A = constvar([ cos(θ) -sin(θ); sin(θ) cos(θ) ])
    
    for i in 1:n
        x[i] ~ MvNormalMeanCovariance(A * x_prev, Q)
        y[i] ~ MvNormalMeanCovariance(x[i], P)
        x_prev = x[i]
    end
    
    return x, y
end

function generate_input(rng, n) 
    θ = π / 35

    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    Q = Matrix(Diagonal(1.0 * ones(2)))
    P = Matrix(Diagonal(1.0 * ones(2)))

    x_prev = [ 10.0, -10.0 ]

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormal(A * x_prev, Q))
        y[i] = rand(rng, MvNormal(x[i], Q))
        
        x_prev = x[i]
    end

    return y
end

function benchmark(input)
    n = length(input)

    _, (x, y) = lgssm(n);

    xbuffer   = buffer(Marginal, n)
    marginals = getmarginals(x)

    subscription = subscribe!(marginals, xbuffer)
    
    update!(y, observations)

    unsubscribe!(subscription)
    
    return getvalues(xbuffer)
end

end