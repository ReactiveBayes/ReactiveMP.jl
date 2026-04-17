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
ReactiveMP.getannotations(marginal::Marginal)
ReactiveMP.as_marginal
ReactiveMP.to_marginal
```

```@example marginal
using ReactiveMP, BayesBase, ExponentialFamily

distribution  = ExponentialFamily.NormalMeanPrecision(0.0, 1.0)
marginal      = Marginal(distribution, false, true)
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

## Marginal observable

Within the reactive message passing framework, marginals are not computed once and stored as values — instead they live as *streams* that continuously emit updated beliefs as new messages arrive. `MarginalObservable` is the container for such a stream.

```@docs
ReactiveMP.MarginalObservable
```

Every [`ReactiveMP.AbstractVariable`](@ref) holds one `MarginalObservable`, accessed via [`ReactiveMP.get_stream_of_marginals`](@ref). The observable starts *unconnected*: its internal `LazyObservable` has no upstream source until the factor graph is activated. During activation, `ReactiveMP.connect!` wires the lazy stream to a computed source (e.g. `collectLatest` over inbound messages for a [`ReactiveMP.RandomVariable`](@ref), or the observation channel for a [`ReactiveMP.DataVariable`](@ref)). After that point, every message update propagates through the graph and the `MarginalObservable` emits a fresh `Marginal`.

The internal `RecentSubject` ensures that:
- any subscriber that joins after the first emission immediately receives the current belief via `Rocket.getrecent`
- [`ReactiveMP.set_initial_marginal!`](@ref) can seed an initial value *before* activation, so that rules which depend on a marginal at iteration zero have something to read

All downstream subscriptions go through the `LazyObservable`, not the subject directly, so they see the full computed stream rather than only manually pushed values.
