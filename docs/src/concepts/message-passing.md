# [Message passing](@id concepts-message-passing)

Message passing is the algorithm that ReactiveMP.jl uses to perform inference on a [factor graph](@ref concepts-factor-graphs). Instead of computing the full joint distribution, each factor node and each variable node exchange small, local summaries — called **messages** — with their neighbors. The posterior beliefs emerge from combining these messages.

## [Belief propagation](@id concepts-message-passing-bp)

**Belief propagation** (also known as the sum-product algorithm) computes *exact* marginal posteriors on tree-shaped graphs. The key idea is that a message from a factor node `f` toward a variable `x` summarizes everything `f` knows about `x` from the rest of the graph:

```math
\mu_{f \to x}(x) = \int f(x, y, z) \; \mu_{y \to f}(y) \; \mu_{z \to f}(z) \; \mathrm{d}y \; \mathrm{d}z
```

The message from `x` back toward `f` collects the beliefs arriving at `x` from all *other* connected factors. The marginal `q(x)` is then the product of all incoming messages at `x`.

On graphs with cycles, this same procedure is run iteratively (loopy belief propagation) and typically converges to a good approximation.

## [Variational message passing](@id concepts-message-passing-vmp)

**Variational message passing** (VMP) is a generalization that performs approximate inference by minimizing the Bethe free energy — a variational objective — rather than computing exact integrals. ReactiveMP.jl implements VMP as the primary inference algorithm because:

1. It includes exact belief propagation as a special case (no factorization constraints = exact BP).
2. It handles non-conjugate and complex models via the **mean-field** or **structured factorization** assumptions.
3. It admits a local, message-level implementation that fits the reactive computation model naturally.

Under a mean-field factorization assumption `q(x, y) = q(x) q(y)`, the VMP message from factor `f` toward variable `x` becomes:

```math
\mu_{f \to x}(x) = \exp \int q(y) \, q(z) \log f(x, y, z) \; \mathrm{d}y \; \mathrm{d}z
```

Notice that this uses *marginals* `q(y)` and `q(z)` rather than messages `μ(y)` and `μ(z)`. ReactiveMP.jl tracks this distinction through its [functional dependencies pipeline](@ref lib-node-functional-dependencies-pipeline).

For a deeper treatment of the theory, see the [PhD dissertation](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) that ReactiveMP.jl is based on.

## [How ReactiveMP.jl chooses the algorithm](@id concepts-message-passing-dispatch)

ReactiveMP.jl does not ask you to pick an algorithm up front. Instead, the correct message update rule is selected automatically based on:

1. **The node type** ([`Stochastic`](@ref) or [`Deterministic`](@ref)) — deterministic nodes always use BP-style messages.
2. **The factorization assumption** attached to the model — mean-field or structured factorization triggers the appropriate VMP rule.
3. **Julia's multiple dispatch** — `@rule` definitions are dispatched on the node type, the outgoing edge, and the types of incoming messages/marginals.

This means adding a new factorization assumption automatically routes computation to the right rules without changing any node code.

## [The reactive computation model](@id concepts-message-passing-reactive)

The word *reactive* in the package name refers to how messages are scheduled. Many message passing libraries build an explicit computation schedule (e.g., forward-backward passes) before inference starts. ReactiveMP.jl takes a different approach: **there is no pre-built schedule**. Instead:

- Each variable and factor node holds a *reactive stream* (a [`ReactiveMP.MessageObservable`](@ref) or [`ReactiveMP.MarginalObservable`](@ref)) that emits updated values whenever its inputs change.
- When new data arrives via [`new_observation!`](@ref), the change propagates automatically through the graph, triggering only the rules that depend on the updated value.
- The propagation order is determined by the graph structure at runtime, not a static plan.

## [The reactive computation model](@id concepts-message-passing-reactive)

The word *reactive* in the package name refers to how messages are scheduled. Many message passing libraries build an explicit computation schedule (e.g., forward-backward passes) before inference starts. ReactiveMP.jl takes a different approach: **there is no pre-built schedule**. Instead:

- Each variable and factor node holds a *reactive stream* (a [`ReactiveMP.MessageObservable`](@ref) or [`ReactiveMP.MarginalObservable`](@ref)) that emits updated values whenever its inputs change.
- When new data arrives via [`new_observation!`](@ref), the change propagates automatically through the graph, triggering only the rules that depend on the updated value.
- The propagation order is determined by the graph structure at runtime, not a static plan.

This reactive design is built on top of [Rocket.jl](https://github.com/ReactiveBayes/Rocket.jl), a Julia library for reactive programming with observables. For a higher-level explanation of this paradigm and how to conceptualize messages as streams, see the [Reactive Programming Model](@ref concepts-reactive-programming). Understanding that messages are *streams* rather than *values* helps explain the activation step described in [Inference lifecycle](@ref concepts-inference-lifecycle).

## [Messages and marginals](@id concepts-message-passing-types)

ReactiveMP.jl uses two distinct wrapper types:

- [`Message`](@ref) — a message flowing along a single edge, from a factor toward a variable (or vice versa).
- [`Marginal`](@ref) — the posterior belief at a variable, computed as the normalized product of all incoming messages.

Both are thin wrappers around a probability distribution object. The separation allows the engine to track metadata such as whether a value is clamped (fixed) or initial (a prior seed), and to carry optional annotations for model evidence computation (see [Annotations](@ref lib-annotations)).

## [Next steps](@id concepts-message-passing-next)

- [Messages](@ref lib-message) — detailed description of the `Message` type and message observables.
- [Marginals](@ref lib-marginal) — the `Marginal` type and marginal observables.
- [Message update rules](@ref lib-rules) — how to define and query rules with `@rule` and `@marginalrule`.
- [Inference lifecycle](@ref concepts-inference-lifecycle) — the three phases of construction, activation, and observation.
