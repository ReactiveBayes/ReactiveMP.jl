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

Next step is to define an inference procedure.

```@example linreg
function inference(xdata, ydata)
    @assert length(xdata) == length(ydata)
    
    n = length(xdata)
    
    model, (a, b, x, y) = linear_regression(n)
    
    as = storage(Marginal)
    bs = storage(Marginal)
    
    asub = subscribe!(getmarginal(a), as)
    bsub = subscribe!(getmarginal(b), bs)
    
    # We need to initialise messages for loopy belief propagation
    setmessage!(b, NormalMeanVariance(0.0, 100.0))
    
    # loopy belief propagation
    for i in 1:20
        update!(x, xdata)
        update!(y, ydata)
    end
    
    return getvalues(as), getvalues(bs)
end
```

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

```@example linreg
ares, bres = inference(xdata, ydata)
nothing #hide
```

```@example lingreg
println("Real a: ", reala, " | Estimated a: ", mean(ares), " | Error: ", abs(mean(ares) - reala))
println("Real b: ", realb, " | Estimated b: ", mean(bres), " | Error: ", abs(mean(bres) - realb))
nothing #hide
```

We can see that ReactiveMP.jl estimated real values of linear regression coefficients with high precision. Lets also test the benchmark of the resulting inference procedure.

```@example linreg
@benchmark inference($xdata, $ydata)
```





