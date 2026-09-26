# Linearization

Local linearization replaces a function by its first-order Taylor expansion at the inputs'
means, computed by automatic differentiation. Where the unscented transform returns moments,
linearization returns the linear map `(A, b)`, and the caller propagates the moments through it:
a normal `N(m, V)` becomes `N(A * m + b, A * V * A')`.

```@docs
Linearization
approximate(::Linearization, ::G, ::Tuple) where {G}
local_linearization
```
