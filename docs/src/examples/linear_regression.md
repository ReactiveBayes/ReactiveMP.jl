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
    b ~ NormalMeanVariance(0.0, 1.0)
    
    x = datavar(Float64, n)
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanVariance(x[i] * b + a, 1.0)
    end
    
    return a, b, x, y
end
```

## Dataset

In order to test our inference procedure we create a test dataset where observations are corrupted with gaussian white noise (with known variance).

```@example linreg
reala = 25
realb = 0.5

N = 100

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
    iterations = 20
)
```

```@example linreg
a = results.posteriors[:a] 
```

```@example linreg
b = results.posteriors[:b]
```

```@example linreg
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





