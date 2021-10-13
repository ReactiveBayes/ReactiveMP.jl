# [Example: Linear Gaussian State Space Model](@id examples-linear-gaussian-state-space-model)

In this example the goal is to estimate hidden states of a Linear Dynamical process where all hidden states are Gaussians. A simple multivariate Linear Gaussian State Space Model can be described with the following equations:

```math
\begin{aligned}
 p(x_i|x_{i - 1}) & = \mathcal{N}(x_i|A * x_{i - 1}, \mathcal{P}),\\
 p(y_i|x_i) & = \mathcal{N}(y_i|B * x_i, \mathcal{Q}),
\end{aligned}
```

where ``x_i`` are hidden states, ``y_i`` are noisy observations, ``A``, ``B`` are state transition and observational matrices, ``\mathcal{P}`` and ``\mathcal{Q}`` are state transition noise and observation noise covariance matrices. For a more rigorous introduction to Linear Gaussian Dynamical systems we refer to [Simo Sarkka, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf) book.

To model this process in `ReactiveMP`, first, we start with importing all needed packages:

```@example lgssm
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, LinearAlgebra, Plots
```

Next step, is to generate some synthetic data:

```@example lgssm
function generate_data(rng, A, B, Q, P)
    x_prev = [ 10.0, -10.0 ]

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormal(A * x_prev, Q))
        y[i] = rand(rng, MvNormal(B * x[i], P))
        x_prev = x[i]
    end
    
    return x, y
end
```

```@example lgssm
# Seed for reproducibility
seed = 1234

rng = MersenneTwister(seed)

# We will model 2-dimensional observations with rotation matrix `A`
# To avoid clutter we also assume that matrices `A`, `B`, `P` and `Q`
# are known and fixed for all time-steps
θ = π / 35
A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
B = diageye(2)
Q = diageye(2)
P = 25.0 .* diageye(2)

# Number of observations
n = 300

nothing #hide
```

!!! note
    For large number of observations you will need yo use `limit_stack_depth = 100` option during model creation, e.g. `model, (x, y) = create_model(..., options = (limit_stack_depth = 100, ))`


```@example lgssm
x, y = generate_data(rng, A, B, Q, P)
nothing #hide
```

Lets plot our synthetic dataset. Lines represent our hidden states we want to estimate using noisy observations, which are represented as dots.

```@example lgssm
slicedim(dim) = (a) -> map(e -> e[dim], a)

px = plot()

px = plot!(px, x |> slicedim(1), label = "Hidden Signal (dim-1)", color = :orange)
px = scatter!(px, y |> slicedim(1), label = false, markersize = 2, color = :orange)
px = plot!(px, x |> slicedim(2), label = "Hidden Signal (dim-2)", color = :green)
px = scatter!(px, y |> slicedim(2), label = false, markersize = 2, color = :green)

plot(px)
```

To create a model we use `GraphPPL` package and `@model` macro:

```@example lgssm
@model function rotate_ssm(n, x0, A, B, Q, P)
    
    # We create constvar references for better efficiency
    cA = constvar(A)
    cB = constvar(B)
    cQ = constvar(Q)
    cP = constvar(P)
    
    # `x` is a sequence of hidden states
    x = randomvar(n)
    # `y` is a sequence of "clamped" observations
    y = datavar(Vector{Float64}, n)
    
    x_prior ~ MvNormalMeanCovariance(mean(x0), cov(x0))
    x_prev = x_prior
    
    for i in 1:n
        x[i] ~ MvNormalMeanCovariance(cA * x_prev, cQ)
        y[i] ~ MvNormalMeanCovariance(cB * x[i], cP)
        x_prev = x[i]
    end
    
    return x, y
end
```

Also for convenience we create an `inference` function to infer hidden states of our system:

```@example lgssm
function inference(data, x0, A, B, Q, P)

    # We create a model and get references for 
    # hidden states and observations
    model, (x, y) = rotate_ssm(n, x0, A, B, Q, P);

    xbuffer   = buffer(Marginal, n)
    bfe       = nothing
    
    # We subscribe on posterior marginals of `x`
    xsubscription = subscribe!(getmarginals(x), xbuffer)
    # We are also intereset in BetheFreeEnergy functional,
    # which in this case is equal to minus log evidence
    fsubcription = subscribe!(score(BetheFreeEnergy(), model), (v) -> bfe = v)

    # `update!` updates our clamped datavars
    update!(y, data)

    # It is important to always unsubscribe
    unsubscribe!((xsubscription, fsubcription))
    
    return xbuffer, bfe
end
```

To run inference we also specify prior for out first time-step:

```@example lgssm
x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
nothing # hide
```

```@example lgssm
xmarginals, bfe = inference(y, x0, A, B, Q, P)
nothing #hide
```

```@example lgssm
px = plot()

px = plot!(px, x |> slicedim(1), label = "Hidden Signal (dim-1)", color = :orange)
px = plot!(px, x |> slicedim(2), label = "Hidden Signal (dim-2)", color = :green)

px = plot!(px, mean.(xmarginals) |> slicedim(1), ribbon = var.(xmarginals) |> slicedim(1) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal (dim-1)", color = :teal)
px = plot!(px, mean.(xmarginals) |> slicedim(2), ribbon = var.(xmarginals) |> slicedim(2) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal (dim-1)", color = :violet)

plot(px)
```

As we can see from our plot, estimated signal resembles closely to the real hidden states with small variance. We maybe also interested in the value for minus log evidence:

```@example lgssm
bfe
```

We may be also interested in performance of our resulting Belief Propagation algorithm:

```@example lgssm
@benchmark inference($y, $x0, $A, $B, $Q, $P)
```