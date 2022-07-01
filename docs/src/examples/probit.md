# [Example: Expectation Propagation / Probit model] (@id examples-probit)

ReactiveMP comes with support for expectation propagation (EP). In this demo we illustrate EP in the context of state-estimation in a linear state-space model that combines a Gaussian state-evolution model with a discrete observation model. Here, the probit function links continuous variable $x_t$ with the discrete variable $y_t$. The model is defined as:

```math
\begin{align}
    u &= 0.1 \\
    x_0 &\sim \mathcal{N}(0, 100) \\
    x_t &\sim \mathcal{N}(x_{t-1}+ u, 0.01) \\
    y_t &\sim \mathrm{Ber}(\Phi(x_t))
\end{align}
```

## Import packages

```@example ep
using StatsFuns: normcdf
using Random, Plots

using ReactiveMP, GraphPPL, Rocket, StableRNGs
```

## Data generation process

```@example ep
function generate_data(nr_samples::Int64; seed = 123)
    
    rng = StableRNG(seed)
    
    # hyper parameters
    u = 0.1

    # allocate space for data
    data_x = zeros(nr_samples + 1)
    data_y = zeros(nr_samples)
    
    # initialize data
    data_x[1] = -2
    
    # generate data
    for k = 2:nr_samples + 1
        
        # calculate new x
        data_x[k] = data_x[k-1] + u + sqrt(0.01)*randn(rng)
        
        # calculate y
        data_y[k-1] = normcdf(data_x[k]) > rand(rng)
        
    end
    
    # return data
    return data_x, data_y
    
end
```

```@example ep
n = 40

data_x, data_y = generate_data(n)

p = plot(xlabel = "t", ylabel = "x, y")
p = scatter!(p, data_y, label = "y")
p = plot!(p, data_x[2:end], label = "x")
```

## Model specification

```@example ep
@model function probit_model(nr_samples::Int64)
    
    # allocate space for variables
    x = randomvar(nr_samples + 1)
    y = datavar(Float64, nr_samples)
    
    # specify uninformative prior
    x[1] ~ NormalMeanPrecision(0.0, 0.01)
    
    # create model 
    for k = 2:nr_samples + 1
        x[k] ~ NormalMeanPrecision(x[k - 1] + 0.1, 100)
        y[k - 1] ~ Probit(x[k]) where {
            # Probit node by default uses RequireInbound pipeline with vague(NormalMeanPrecision) message as initial value for `in` edge
            # To change initial value use may specify it manually, like. Changes to the initial message may improve stability in some situations
            pipeline = RequireInbound(in = NormalMeanPrecision(0, 0.01)) 
        }
    end
    
    # return parameters
    return x, y
    
end
```

## Inference

```@example ep
result = inference(
    model = Model(probit_model, length(data_y)), 
    data  = (
        y = data_y, 
    ), 
    iterations = 10, 
    returnvars = (
        x = KeepLast(),
    ),
    showprogress = true,
    free_energy  = true
)
```

```@example ep
mx = result.posteriors[:x]

p = plot(xlabel = "t", ylabel = "x, y", legend = :bottomright)
p = scatter!(p, data_y, label = "y")
p = plot!(p, data_x[2:end], label = "x", lw = 2)
p = plot!(mean.(mx)[2:end], ribbon = std.(mx)[2:end], fillalpha = 0.2, label="x (inferred mean)")

f = plot(xlabel = "t", ylabel = "BFE")
f = plot!(result.free_energy[2:end], label = "Bethe Free Energy")

plot(p, f, size = (800, 400))
```
