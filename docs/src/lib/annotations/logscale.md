# [Log-scale annotations](@id lib-annotations-logscale)

## Background: scale factors in message passing

In sum-product message passing on a Forney-style factor graph, a message ``\vec{\mu}_{s_j}(s_j)`` flowing along an edge is in general *unnormalised*. It can be decomposed as

```math
\vec{\mu}_{s_j}(s_j) = \beta_{s_j} \cdot \hat{p}_{s_j}(s_j),
```

where ``\hat{p}_{s_j}(s_j)`` is the normalised probability distribution (what ReactiveMP stores as the message's data) and ``\beta_{s_j}`` is the **scale factor** — a positive scalar that carries the accumulated normalisation constant of the message.

A key result from [van Erp et al. (2023)](https://arxiv.org/abs/2306.05965) is that in an acyclic graph the product of two colliding messages integrates to exactly the model evidence:

```math
\int \vec{\mu}_{s_j}(s_j)\, \overleftarrow{\mu}_{s_j}(s_j)\, \mathrm{d}s_j = p(y = \hat{y}).
```

This means the model evidence can be read off locally at *any* edge by tracking the scale factor. This enables **Bayesian model comparison** — averaging, selection, and combination — to be performed automatically as part of the same message-passing run that computes posteriors, without any separate evidence computation.

## Log-scale factors

For numerical stability, ReactiveMP tracks the *logarithm* of the scale factor, ``\log \beta``. When two messages are multiplied to form a product message, the log-scale of the result is:

```math
\log \beta_\text{new} = \log \beta_\text{left} + \log \beta_\text{right} + \texttt{compute\_logscale}(\hat{p}_\text{new},\, \hat{p}_\text{left},\, \hat{p}_\text{right}),
```

where ``\texttt{compute\_logscale}(\hat{p}_\text{new}, \hat{p}_\text{left}, \hat{p}_\text{right})`` is the log of the normalisation constant of the product:

```math
\texttt{compute\_logscale} = \log Z = \log \int \hat{p}_\text{left}(x)\, \hat{p}_\text{right}(x)\, \mathrm{d}x.
```

This function is defined in [`BayesBase.jl`](https://github.com/ReactiveBayes/BayesBase.jl) and extended for specific distribution families in [`ExponentialFamily.jl`](https://github.com/ReactiveBayes/ExponentialFamily.jl).

## When is the log-scale zero?

The log-scale ``\log \beta`` is zero exactly when the rule's raw factor product already integrates to 1 — i.e. the rule only reparameterises its inputs without dividing out a normalisation constant. This is the case for most **conjugate continuous rules**, for example a Normal message rule that takes a Normal prior and a PointMass precision and returns a new Normal: the computation is a direct parameter transformation and nothing is normalised away.

The log-scale is **non-zero** whenever the rule involves a factor that does not integrate to 1 over the variable being messaged to. Two important classes:

- **Discrete observed nodes** — for example, a Bernoulli node with observed output ``y = 1`` and unknown ``p``. The raw factor is ``f(p) = p``, which integrates to ``\tfrac{1}{2}`` over ``[0,1]``. The normalised message representation is ``\mathrm{Beta}(2, 1)``, but the lost constant ``\tfrac{1}{2}`` must be recorded: ``\log \beta = -\log 2``.

- **Mixture and categorical nodes** — the backward message toward the model selection variable ``m`` contains the model evidence of each component as its scale factor (see equation 42 in van Erp et al. (2023)). This is the mechanism that makes automated Bayesian model comparison possible.

Critically, **the normalised output distribution alone does not reveal the log-scale**. Both a zero-logscale rule and a non-zero-logscale rule return a properly normalised distribution object — looking at `Beta(2,1)` does not tell you that `log β = -log 2` was lost. This is why `@logscale` must be set explicitly by the rule author.

## Inside rule bodies: `@logscale`

When a message update rule computes a message whose normalisation constant is known analytically, it records the log-scale factor using the `@logscale` macro:

```julia
@rule NormalMeanVariance(:out, Marginalisation) (m_μ::UnivariateNormalDistributionsFamily, m_σ²::PointMass) = begin
    @logscale 0   # conjugate reparameterisation — no normalisation constant is lost
    return NormalMeanVariance(mean(m_μ), var(m_μ) + mean(m_σ²))
end
```

For rules where a non-trivial normalisation constant is divided out, the exact value must be provided:

```julia
@rule Bernoulli(:p, Marginalisation) (m_out::PointMass,) = begin
    @logscale log(mean(m_out))   # log-likelihood of the observed value
    return Beta(...)
end
```

If a rule does not call `@logscale` and `LogScaleAnnotations` is active, ReactiveMP applies a fallback: if all incoming messages and marginals are `PointMass` distributions (i.e. the node is deterministic given its inputs) the log-scale is set to zero. In all other cases an error is raised to prevent silently wrong model evidence computations.

## API

```@docs
ReactiveMP.LogScaleAnnotations
ReactiveMP.getlogscale
ReactiveMP.@logscale
```

## References

- van Erp, B., Nuijten, W. W. L., van de Laar, T., & de Vries, B. (2023). *Automating Model Comparison in Factor Graphs*. Entropy, 25(8), 1138. [https://doi.org/10.3390/e25081138](https://doi.org/10.3390/e25081138)
