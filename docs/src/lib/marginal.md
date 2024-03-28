# [Marginal implementation](@id lib-marginal)

## Marginal type

All marginals are encoded with the type `Marginal`. 

```@docs
Marginal
```

From an implementation point a view the `Marginal` structure does nothing but hold some `data` object and redirects most of the statistical related functions to that `data` object. However, this object is used extensively in Julia's multiple dispatch. 

```@docs
ReactiveMP.getdata(marginal::Marginal)
ReactiveMP.is_clamped(marginal::Marginal)
ReactiveMP.is_initial(marginal::Marginal)
ReactiveMP.getaddons(marginal::Marginal)
ReactiveMP.as_marginal
```

```@example marginal
using ReactiveMP, BayesBase, ExponentialFamily

distribution  = ExponentialFamily.NormalMeanPrecision(0.0, 1.0)
marginal      = Marginal(distribution, false, true, nothing)
```

```@example marginal
mean(marginal), precision(marginal)
```

```@example marginal
logpdf(marginal, 1.0)
```

```@example marginal
is_clamped(marginal), is_initial(marginal)
```