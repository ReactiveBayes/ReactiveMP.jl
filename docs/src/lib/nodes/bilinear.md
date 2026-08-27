# [Bilinear node](@id lib-nodes-bilinear)

The `Bilinear` node encodes the stochastic bilinear interaction

```math
\phi(out, in, a) = \exp(out \cdot a \cdot in)
```

where the coupling coefficient `a` is expected to carry a `PointMass` marginal. The node is intended to be used with the structured factorization `q(out, in) q(a)`: the potential couples `out` and `in` through the cross-term ``a \cdot out \cdot in`` only, so the joint marginal `q(out, in)` is a bivariate Gaussian whenever the incoming messages on `out` and `in` are Gaussian.

The potential is not integrable on its own: the individual messages towards `out` and `in` are improper Gaussians (negative precision), and the posteriors are proper only when the precisions of the incoming messages dominate the coupling, i.e. ``w_{out} w_{in} > a^2``. Note that this factor is exactly the cross-term of a Gaussian density, since ``\mathcal{N}(out; in, w^{-1}) \propto \exp(-w \cdot out^2/2) \exp(w \cdot out \cdot in) \exp(-w \cdot in^2/2)``.

## [Interfaces](@id lib-nodes-bilinear-interfaces)

| Interface | Role |
|-----------|------|
| `out` | First interaction variable |
| `in` | Second interaction variable |
| `a` | Coupling coefficient, expected to be a `PointMass` |

```@docs
ReactiveMP.Bilinear
```
