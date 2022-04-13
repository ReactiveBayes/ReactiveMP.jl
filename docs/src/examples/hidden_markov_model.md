# [Example: Hidden Markov Model](@id examples-hidden-markov-model)

In this demo we are interested in Bayesian inference of parameters of a hidden Markov model (HMM).,
Specifically, we consider a first-order HMM with hidden states $s_0, s_1, \dots, s_T$ and observations $x_1, \dots, x_T$ governed by a state transition probability matrix $A$ and an observation probability matrix $B$:,

```math
\begin{align*}
    s_t & \sim \mathcal{C}at(A s_{t-1}),\\
    x_t & \sim \mathcal{C}at(B s_t).\\,
\end{align*}
```


We assume three possible states (\"red\", \"green\" and \"blue\"), and the goal is to estimate matrices $A$ and $B$ from a simulated data set. To have a full Bayesian treatment of the problem, both $A$ and $B$ are endowed with priors (Dirichlet distributions on the columns)."

```@example hmm
using Rocket, ReactiveMP, GraphPPL
using Random, BenchmarkTools, Distributions, MacroTools, LinearAlgebra
using Plots
```

```@example hmm
function rand_vec(rng, distribution::Categorical) 
    k = ncategories(distribution)
    s = zeros(k)
    s[ rand(rng, distribution) ] = 1.0
    s
end

function generate_data(n_samples; seed = 124)
    
    rng = MersenneTwister(seed)
    
    # Transition probabilities (some transitions are impossible)
    A = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9] 
    # Observation noise
    B = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9] 
    # Initial state
    s_0 = [1.0, 0.0, 0.0] 
    # Generate some data
    s = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the states
    x = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the observations
    
    s_prev = s_0
    
    for t = 1:n_samples
        a = A * s_prev
        s[t] = rand_vec(rng, Categorical(a ./ sum(a)))
        b = B * s[t]
        x[t] = rand_vec(rng, Categorical(b ./ sum(b)))
        s_prev = s[t]
    end
    
    return x, s
end
```

```@example hmm
# Test data
N = 100

x_data, s_data = generate_data(N)

scatter(argmax.(s_data))
```

## Model specification

```@example hmm
# Model specification
@model function hidden_markov_model(n)
    
    A ~ MatrixDirichlet(ones(3, 3)) 
    B ~ MatrixDirichlet([ 10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0 ])
    
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    
    s = randomvar(n)
    x = datavar(Vector{Float64}, n)
    
    s_prev = s_0
    
    for t in 1:n
        s[t] ~ Transition(s_prev, A) 
        x[t] ~ Transition(s[t], B)
        s_prev = s[t]
    end
    
end

@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end
```

## Inference 

```@example hmm
idata = (x = x_data, )

imodel = Model(hidden_markov_model, N)

imarginals = (
    A = vague(MatrixDirichlet, 3, 3), 
    B = vague(MatrixDirichlet, 3, 3), 
    s = vague(Categorical, 3)
)

ireturnvars = (
    A = KeepLast(),
    B = KeepLast(),
    s = KeepLast()
)

result = inference(
    model         = imodel, 
    data          = idata,
    constraints   = hidden_markov_model_constraints(),
    initmarginals = imarginals, 
    returnvars    = ireturnvars, 
    iterations    = 20, 
    free_energy   = true
)
```

## Results

```@example hmm
mean(result.posteriors[:A])
```

```@example hmm
mean(result.posteriors[:B])
```

```@example hmm
p1 = scatter(argmax.(s_data), title="Inference results", label = "real", ms = 6)
p1 = scatter!(p1, argmax.(ReactiveMP.probvec.(result.posteriors[:s])), label = "inferred", ms = 2)
p2 = plot(result.free_energy, label="Free energy")

plot(p1, p2, layout = @layout([ a; b ]))
```

## Custom inference

```@example hmm
@model [ default_factorisation = MeanField() ] function custom_optimised_hidden_markov_model(n)
    
    A ~ MatrixDirichlet(ones(3, 3)) 
    B ~ MatrixDirichlet([ 10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0 ])
    
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    
    s = randomvar(n)
    x = datavar(Vector{Float64}, n)
    
    s_prev = s_0
    
    for t in 1:n
        s[t] ~ Transition(s_prev, A) where { q = q(out, in)q(a) }
        x[t] ~ Transition(s[t], B)
        s_prev = s[t]
    end
    
    return s, x, A, B
end
```

```@example hmm
function custom_optimised_inference(data, vmp_iters)
    n = length(data)
    
    model, (s, x, A, B) = custom_optimised_hidden_markov_model(model_options(limit_stack_depth = 500), n)
    
    sbuffer = keep(Vector{Marginal})
    Abuffer = keep(Marginal)
    Bbuffer = keep(Marginal)
    fe      = ScoreActor(Float64)
    
    ssub  = subscribe!(getmarginals(s), sbuffer)
    Asub  = subscribe!(getmarginal(A), Abuffer)
    Bsub  = subscribe!(getmarginal(B), Bbuffer)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
    
    setmarginal!(A, vague(MatrixDirichlet, 3, 3))
    setmarginal!(B, vague(MatrixDirichlet, 3, 3))
    
    foreach(s) do svar
        setmarginal!(svar, vague(Categorical, 3))
    end
    
    for i in 1:vmp_iters
        update!(x, data)
    end
    
    unsubscribe!(ssub)
    unsubscribe!(Asub)
    unsubscribe!(Bsub)
    unsubscribe!(fesub)
    
    return map(getvalues, (sbuffer, Abuffer, Bbuffer, fe))
end
```

```@example hmm
sbuffer, Abuffer, Bbuffer, fe = custom_optimised_inference(x_data, 20)

@assert mean.(last(sbuffer)) ≈ mean.(result.posteriors[:s])
nothing #hide
```

```@example hmm
p1 = scatter(argmax.(s_data), title="Inference results", label = "real", ms = 6)
p1 = scatter!(p1, argmax.(ReactiveMP.probvec.(last(sbuffer))), label = "inferred", ms = 2)
p2 = plot(result.free_energy, label="Free energy")

plot(p1, p2, layout = @layout([ a; b ]))
```

## Benchmark timings

```@example hmm
@benchmark inference(
    model         = $imodel, 
    data          = $idata,
    constraints   = hidden_markov_model_constraints(),
    initmarginals = $imarginals, 
    returnvars    = $ireturnvars, 
    iterations    = 20, 
    free_energy   = true
)
```

```@example hmm
@benchmark custom_optimised_inference($x_data, 20)
```

