# MessagePassingRulesApproximations

Numerics for propagating Gaussian moments through a function, and for expectations under a
normal: the unscented transform, local linearization, Gauss–Hermite cubature and a
Rauch–Tung–Striebel smoother. It works on means and covariances, not on distributions, and knows
nothing of message passing; node packages build their rules on it.

```julia
using MessagePassingRulesApproximations

m, V = approximate(Unscented(), x -> 2x + 1, (1.0,), (0.5,))   # ≈ (3.0, 2.0)
A, b = approximate(Linearization(), x -> x^2, (3.0,))          # (6.0, -9.0)
```

- Documentation: `make docs-approximations` from the repository root builds it into
  `docs/build`; it will be published at <https://reactivebayes.github.io/MessagePassingRulesApproximations.jl/dev/>.
- Tests: `make test-approximations`.
- Depends on LinearAlgebra, FastCholesky, FastGaussQuadrature and ForwardDiff only, on no
  distribution package. Julia 1.11 or later. MIT licence.
