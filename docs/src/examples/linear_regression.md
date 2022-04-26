# [Example: Linear Regression](@id examples-linear-regression)

In this example we are going to perform a simple linear regression problem, but in the Bayesian setting. We specify the model:
```math
y_i = a * x_i + b
```
where $a$ and $b$ are random variables with some vague priors.

First step is to import all needed packages and define the model:

```@example linreg
using ReactiveMP, GraphPPL, Rocket, Random, Plots, StableRNGs, BenchmarkTools
```

## Model specification

```@example linreg
@model function linear_regression(n)
    a ~ NormalMeanVariance(0.0, 1.0)
    b ~ NormalMeanVariance(0.0, 100.0)
    
    x = datavar(Float64, n)
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanVariance(a * x[i] + b, 1.0)
    end
    
    return a, b, x, y
end
```

## Dataset

In order to test our inference procedure we create a test dataset where observations are corrupted with gaussian white noise (with known variance).

```@example linreg
reala = 25
realb = 0.5

N = 250

rng = StableRNG(1234)

xorig = collect(1:N)

xdata = xorig .+ randn(rng, N)
ydata = reala .+ realb .* xorig .+ randn(rng, N)

plot(xdata, label = "X", title = "Linear regression dataset")
plot!(ydata, label = "Y")
```

## Inference

```@example linreg
results = inference(
    model = Model(linear_regression, length(xdata)), 
    data  = (y = ydata, x = xdata), 
    initmessages = (b = NormalMeanVariance(0.0, 100.0), ), 
    returnvars   = (a = KeepLast(), b = KeepLast()), 
    iterations   = 20
);
```

```@example linreg
pra = plot(range(-3, 3, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 1.0), x), title="Prior for a parameter", fillalpha=0.3, fillrange = 0, label="Prior P(a)", c=1,)
pra = vline!(pra, [ reala ], label="Real a", c = 3)
psa = plot(range(0.45, 0.55, length = 1000), (x) -> pdf(results.posteriors[:a], x), title="Posterior for a parameter", fillalpha=0.3, fillrange = 0, label="Posterior P(a)", c=2,)
psa = vline!(psa, [ reala ], label="Real a", c = 3)

plot(pra, psa, size = (1000, 200))
```

```@example linreg
prb = plot(range(-40, 40, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 100.0), x), title="Prior for b parameter", fillalpha=0.3, fillrange = 0, label="Prior P(b)", c=1, legend = :topleft)
prb = vline!(prb, [ realb ], label="Real b", c = 3)
psb = plot(range(23, 28, length = 1000), (x) -> pdf(results.posteriors[:b], x), title="Posterior for b parameter", fillalpha=0.3, fillrange = 0, label="Posterior P(b)", c=2, legend = :topleft)
psb = vline!(psb, [ realb ], label="Real b", c = 3)

plot(prb, psb, size = (1000, 200))
```

```@example linreg
a = results.posteriors[:a]
b = results.posteriors[:b]

println("Real a: ", reala, " | Estimated a: ", mean(a), " | Error: ", abs(mean(a) - reala))
println("Real b: ", realb, " | Estimated b: ", mean(b), " | Error: ", abs(mean(b) - realb))
nothing #hide
```

We can see that ReactiveMP.jl estimated real values of linear regression coefficients with high precision. Lets also test the benchmark of the resulting inference procedure.

```@example linreg
@benchmark inference(
    model = Model($linear_regression, length($xdata)), 
    data  = (y = $ydata, x = $xdata), 
    initmessages = (b = NormalMeanVariance(0.0, 100.0), ), 
    returnvars   = (a = KeepLast(), b = KeepLast()), 
    iterations = 20
)
```





