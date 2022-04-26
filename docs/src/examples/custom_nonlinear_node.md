# [Example: Custom nonlinear node] (@id examples-custom-nonlinear-node)

```@example nonlinear
using ReactiveMP, Distributions, Random, BenchmarkTools, Rocket, GraphPPL, StableRNGs
```

Here is an example of creating custom node with nonlinear function approximation with samplelist.

## Custom node structure

```@example nonlinear
struct NonlinearNode end # Dummy structure just to make Julia happy

struct NonlinearMeta{R, F}
    rng      :: R
    fn       :: F   # Nonlinear function, we assume 1 float input - 1 float ouput
    nsamples :: Int # Number of samples used in approximation
end
```

```@example nonlinear
@node NonlinearNode Deterministic [ out, in ]
```

We need to define two Sum-product message computation rules for our new custom node
- Rule for outbound message on `out` edge given inbound message on `in` edge
- Rule for outbound message on `in` edge given inbound message on `out` edge
- Both rules accept optional meta object

```@example nonlinear
# Rule for outbound message on `out` edge given inbound message on `in` edge
@rule NonlinearNode(:out, Marginalisation) (m_in::NormalMeanVariance, meta::NonlinearMeta) = begin 
    samples = rand(meta.rng, m_in, meta.nsamples)
    return SampleList(map(meta.fn, samples))
end

# Rule for outbound message on `in` edge given inbound message on `out` edge
@rule NonlinearNode(:in, Marginalisation) (m_out::Gamma, meta::NonlinearMeta) = begin     
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_out, meta.fn(x)))
end
```

## Model specification

After we have defined our custom node with custom rules we may proceed with a model specification:

```math
\begin{equation}
\begin{aligned}
p(\theta) &= \mathcal{N}(\theta|\mu_{\theta}, \sigma_{\theta}),\\
p(m) &= \mathcal{N}(\theta|\mu_{m}, \sigma_{m}),\\
p(w) &= f(\theta),\\
p(y_i|m, w) &= \mathcal{N}(y_i|m, w),
\end{aligned}
\end{equation}
```

Given this IID model, we aim to estimate the precision of a Gaussian distribution. We pass a random variable $\theta$ through a non-linear transformation $f$ to make it positive and suitable for a precision parameter of a Gaussian distribution. We, later on, will estimate the posterior of $\theta$. 

```@example nonlinear
@model function nonlinear_estimation(n)
    
    θ ~ NormalMeanVariance(0.0, 100.0)
    m ~ NormalMeanVariance(0.0, 1.0)
    
    w ~ NonlinearNode(θ)
    
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanPrecision(m, w)
    end
    
    return θ, m, w, y
end
```

```@example nonlinear 
@constraints function nconstsraints(nsamples)
    q(θ) :: SampleList(nsamples, LeftProposal())
    q(w) :: SampleList(nsamples, RightProposal())
    
    q(θ, w, m) = q(θ)q(m)q(w)
end
```

```@example nonlinear
# TODO: check
@meta function nmeta(fn, nsamples)
    NonlinearNode(θ, w) -> NonlinearMeta(StableRNG(123), fn, nsamples)
end
```

Here we generate some data with some arbitrary nonlinearity for precision parameter:

```@example nonlinear
nonlinear_fn(x) = abs(exp(x) * sin(x))
```

```@example nonlinear
seed = 123
rng  = MersenneTwister(seed)

niters   = 15 # Number of VMP iterations
nsamples = 5_000 # Number of samples in approximation

n = 500 # Number of IID samples
μ = -10.0
θ = -1.0
w = nonlinear_fn(θ)

data = rand(rng, NormalMeanPrecision(μ, w), n)
nothing #hide
```

## Inference

```@example nonlinear
result = inference(
    model = Model(nonlinear_estimation, n),
    meta =  nmeta(nonlinear_fn, nsamples),
    constraints = nconstsraints(nsamples),
    data = (y = data, ), 
    initmarginals = (m = vague(NormalMeanPrecision), w = vague(Gamma)),
    returnvars = (θ = KeepLast(), ),
    iterations = niters,  
    showprogress = true
)
```

```@example nonlinear 
using Plots, StatsPlots

estimated = Normal(mean_std(result.posteriors[:θ])...)

plot(estimated, title="Posterior for θ", label = "Estimated", legend = :bottomright, fill = true, fillopacity = 0.2, xlim = (-3, 3), ylim = (0, 2))
vline!([ θ ], label = "Real value of θ")
```

## Benchmark

```@example nonlinear
@benchmark inference(
    model = $(Model(nonlinear_estimation, n)),
    meta = $(nmeta(nonlinear_fn, nsamples)),
    constraints = $(nconstsraints(nsamples)),
    data = (y = $data, ), 
    initmarginals = (m = vague(NormalMeanPrecision), w = vague(Gamma)),
    returnvars = (θ = KeepLast(), ),
    iterations = $niters,  
    showprogress = true
)
```