export Bilinear

@doc raw"""
    Bilinear

Stochastic factor node representing the bilinear interaction

```math
\phi(out, in, a) = \exp(out \cdot a \cdot in)
```

The `a` interface is expected to carry a `PointMass` marginal. The node is intended to be
used with the structured factorization `q(out, in) q(a)`: the potential couples `out` and
`in` through the cross-term `a \cdot out \cdot in` only, so the joint marginal
`q(out, in)` is a bivariate Gaussian whenever the incoming messages on `out` and `in`
are Gaussian.

Note that the potential is not integrable on its own: the individual messages towards
`out` and `in` are improper Gaussians (negative precision), and the posteriors are proper
only when the precisions of the incoming messages dominate the coupling, i.e.
``w_{out} w_{in} > a^2``.

# Interfaces
1. `out`
2. `in`
3. `a` — coupling coefficient, expected to be a `PointMass`.
"""
struct Bilinear end

@node Bilinear Stochastic [out, in, a]

@average_energy Bilinear (q_out_in::Any, q_a::Any) = begin
    # U = -E[log φ] = -mean(q_a) * E_{q(out, in)}[out * in]
    m, V = mean_cov(q_out_in)
    return -mean(q_a) * (V[1, 2] + m[1] * m[2])
end
