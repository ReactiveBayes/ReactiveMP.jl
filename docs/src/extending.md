# Extending the functionality

## Adding a new type of node

`ReactiveMP.jl` package exports the `@node` macro to create a simple factor node with fixed number of arguments.

`@node` macro accepts three arguments:
- A functional form of the new node in form of a Julia type, e.g. `Normal` or `typeof(+)`
- A type of node: Stochastic or Deterministic
- A list of node arguments, e.g. `[ out, mean, variance ]`

!!! note
    By convention a list of node arguments should start with `out` 

Examples:

```julia
@node GaussianMeanVariance Stochastic [ out, m, v ]
@node typeof(+) Deterministic [ out, in1, in2 ]
```