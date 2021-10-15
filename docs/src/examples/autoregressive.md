# [Example: Autoregressive model](@id examples-autoregressive)

In this example we are going to perform an automated Variational Bayesian Inference for autoregressive model that can be represented as following:

```math
\begin{aligned}
p(\gamma) &= \mathrm{Gamma}(\gamma|a, b),\\
p(\mathbf{\theta}) &= \mathcal{N}(\mathbf{\theta}|\mathbf{\mu}, \Sigma),\\
p(x_t|\mathbf{x}_{t-1:t-k}) &= \mathcal{N}(x_t|\mathbf{\theta}^{T}\mathbf{x}_{t-1:t-k}, \gamma^{-1}),\\
p(y_t|x_t) &= \mathcal{N}(y_t|x_t, \tau^{-1}),
\end{aligned}
```

where ``x_t`` is a current state of our system, ``\mathbf{x}_{t-1:t-k}`` is a sequence of ``k`` previous states, ``k`` is an order of autoregression process, ``\mathbf{\theta}`` is a vector of transition coefficients, ``\gamma`` is a precision of state transition process, ``y_k`` is a noisy observation of ``x_k`` with precision ``\tau``.

For a more rigorous introduction to Bayesian inference in Autoregressive models we refer to [Albert Podusenko, Message Passing-Based Inference for Time-Varying Autoregressive Models](https://www.mdpi.com/1099-4300/23/6/683).

We start with importing all needed packages:

```@example ar
using Rocket, ReactiveMP, GraphPPL
using Distributions, LinearAlgebra, Random, Plots, BenchmarkTools
```

Lets generate some synthetic dataset, we use a predefined set of coeffcients for $k$ = 5:

```@example ar
# The following coefficients correspond to stable poles
coefs_ar_5 = [0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288]
nothing #hide
```

```@example ar
function generate_ar_data(rng, n, θ, γ, τ)
    order        = length(θ)
    states       = Vector{Vector{Float64}}(undef, n + 3order)
    observations = Vector{Float64}(undef, n + 3order)
    
    γ_std = sqrt(inv(γ))
    τ_std = sqrt(inv(γ))
    
    states[1] = randn(rng, order)
    
    for i in 2:(n + 3order)
        states[i]       = vcat(rand(rng, Normal(dot(θ, states[i - 1]), γ_std)), states[i-1][1:end-1])
        observations[i] = rand(rng, Normal(states[i][1], τ_std))
    end
    
    return states[1+3order:end], observations[1+3order:end]
end
```

```@example ar
# Seed for reproducibility
seed = 123
rng  = MersenneTwister(seed)

# Number of observations in synthetic dataset
n = 500

# AR process parameters
real_γ = 5.0
real_τ = 5.0
real_θ = coefs_ar_5

states, observations = generate_ar_data(rng, n, real_θ, real_γ, real_τ)
nothing #hide
```

Lets plot our synthetic dataset:

```@example ar
plot(first.(states), label = "Hidden states")
scatter!(observations, label = "Observations")
```

Next step is to specify probabilistic model and run inference procedure with `ReactiveMP`. We use `GraphPPL.jl` package to specify probabilistic model and additional constraints for variational Bayesian Inference. We also specify two different models for Multivariate AR with order ``k`` > 1 and for Univariate AR (reduces to simple State-Space-Model) with order ``k`` = 1.

```@example ar
@model [ default_factorisation = MeanField() ] function lar_model(T::Type{ Multivariate }, n, order, c, stype, τ)
    
    # Parameter priors 
    γ  ~ GammaShapeRate(1.0, 1.0) 
    θ  ~ MvNormalMeanPrecision(zeros(order), diageye(order))
    
    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)
    
    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)
    
    # Prior for first state
    x0 ~ MvNormalMeanPrecision(zeros(order), diageye(order))
    
    x_prev = x0
    
    # AR process requires extra meta information
    meta = ARMeta(Multivariate, order, stype)
    
    for i in 1:n
        # Autoregressive node uses structured factorisation assumption between states
        x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = meta }
        y[i] ~ NormalMeanPrecision(dot(ct, x[i]), cτ)
        x_prev = x[i]
    end
    
    return x, y, θ, γ
end
```

```@example ar
@model [ default_factorisation = MeanField() ] function lar_model(T::Type{Univariate}, n, order, c, stype, τ)
    
    # Parameter priors 
    γ  ~ GammaShapeRate(1.0, 1.0)
    θ  ~ NormalMeanPrecision(0.0, 1.0)
    
    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)
    
    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ) 

    # Prior for first state
    x0 ~ NormalMeanPrecision(0.0, 1.0)
    
    x_prev = x0
    
    # AR process requires extra meta information
    meta = ARMeta(Univariate, order, stype)
    
    for i in 1:n
        x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = meta }
        y[i] ~ NormalMeanPrecision(ct * x[i], cτ)
        x_prev = x[i]
    end
    
    return x, y, θ, γ
end
```

We will use different initial marginals depending on type of our AR process

```@example ar
function init_marginals!(::Type{ Multivariate }, order, γ, θ)
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), diageye(order)))
end

function init_marginals!(::Type{ Univariate }, order, γ, θ)
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, NormalMeanPrecision(0.0, 1.0))
end
```

```@example ar
function inference(data, order, artype, stype, niter, τ)
    
    # We build a full graph based on nber of observatios
    n = length(data)
    
    # Depending on the order of AR process `c` is
    # either a nber or a vector
    c = ReactiveMP.ar_unit(artype, order)
    
    # Note that to run inference for huge model it might be necessary to pass extra 
    # options = (limit_stack_depth = 100,) to limit stack depth during recursive inference procedure
    model, (x, y, θ, γ) = lar_model(artype, n, order, c, stype, τ)
    
    # We are going to keep `γ` and `θ` estimates for all VMP iterations
    # But `buffer` only last posterior estimates for a sequence of hidden states `x`
    # We also will keep Bethe Free Energy in `fe`
    γ_buffer = keep(Marginal)
    θ_buffer = keep(Marginal)
    x_buffer = buffer(Marginal, n)
    fe       = keep(Float64)
    
    γsub  = subscribe!(getmarginal(γ), γ_buffer)
    θsub  = subscribe!(getmarginal(θ), θ_buffer)
    xsub  = subscribe!(getmarginals(x), x_buffer)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
    
    init_marginals!(artype, order, γ, θ)
    
    # We update data several times to perform several VMP iterations
    for i in 1:niter
        update!(y, data)
    end
    
    # It is important to unsubscribe from running observables
    unsubscribe!((γsub, θsub, xsub, fesub))
    
    return γ_buffer, θ_buffer, x_buffer, fe
end
```

```@example ar
γ, θ, xs, fe = inference(observations, length(real_θ), Multivariate, ARsafe(), 15, real_τ)
nothing #hide
```

```@example ar
p1 = plot(first.(states), label="Hidden state")
p1 = scatter!(p1, observations, label="Observations")
p1 = plot!(p1, first.(mean.(xs)), ribbon = sqrt.(first.(var.(xs))), label="Inferred states", legend = :bottomright)

p2 = plot(mean.(γ), ribbon = std.(γ), label = "Inferred transition precision", legend = :bottomright)
p2 = plot!([ real_γ ], seriestype = :hline, label = "Real transition precision")

p3 = plot(getvalues(fe), label = "Bethe Free Energy")

plot(p1, p2, p3, layout = @layout([ a; b c ]))
```

Lets also plot a subrange of our results:

```@example ar
subrange = div(n,5):(div(n, 5) + div(n, 5))

plot(subrange, first.(states)[subrange], label="Hidden state")
scatter!(subrange, observations[subrange], label="Observations")
plot!(subrange, first.(mean.(xs))[subrange], ribbon = sqrt.(first.(var.(xs)))[subrange], label="Inferred states", legend = :bottomright)
```

It is also interesting to see where our AR coefficients converge to:

```@example ar
let
    pθ = plot()

    θms = mean.(θ)
    θvs = var.(θ)
    
    l = length(θms)

    edim(e) = (a) -> map(r -> r[e], a)

    for i in 1:length(first(θms))
        pθ = plot!(pθ, θms |> edim(i), ribbon = θvs |> edim(i) .|> sqrt, label = "Estimated θ[$i]")
    end
    
    for i in 1:length(real_θ)
        pθ = plot!(pθ, [ real_θ[i] ], seriestype = :hline, label = "Real θ[$i]")
    end
    
    plot(pθ, legend = :outertopright, size = (800, 300))
end
```

```@example ar
println("$(length(real_θ))-order AR inference Bethe Free Energy: ", last(fe))
nothing #hide
```

We can also run a 1-order AR inference on 5-order AR data:

```@example ar
γ, θ, xs, fe = inference(observations, 1, Univariate, ARsafe(), 15, real_τ)
nothing #hide
```

```@example ar
println("1-order AR inference Bethe Free Energy: ", last(fe))
nothing #hide
```

We can see that, according to final Bethe Free Energy value, in this example 5-order AR process can describe data better than 1-order AR.

We may be also interested in benchmarking our algorithm:

```@example ar
println("Benchmark for n = $n and AR-$(length(real_θ)) inference");
@benchmark inference($observations, length(real_θ), Multivariate, ARsafe(), 15, real_τ)
```