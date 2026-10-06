# Smoothing

A [deterministic node](@extref MessagePassingRulesBase glossary-deterministic-node),
`out = g(in)`, sends forward the moments of `g(in)` under what is known about `in`. When a
[message](@extref MessagePassingRulesBase glossary-message) about `out` arrives from the other
direction, the belief about `in`, its [marginal](@extref MessagePassingRulesBase glossary-marginal),
must be corrected. [`smoothRTS`](@ref) makes that correction with a Rauch–Tung–Striebel (RTS)
step, the smoothing step of a Kalman smoother.

## The method

Either transform gives the forward statistics: the mean ``\tilde{m}`` and covariance
``\tilde{V}`` of `g(in)`, and the cross-covariance ``C`` between `in` and `g(in)`, under a
normal ``N(m_{\mathrm{in}}, V_{\mathrm{in}})`` for `in`. Together they describe `in` and `out` as
jointly normal:

```math
\begin{pmatrix} \mathrm{in} \\ \mathrm{out} \end{pmatrix} \sim
N\left(\begin{pmatrix} m_{\mathrm{in}} \\ \tilde{m} \end{pmatrix},
\begin{pmatrix} V_{\mathrm{in}} & C \\ C^\top & \tilde{V} \end{pmatrix}\right).
```

A backward normal ``N(m_{\mathrm{bw}}, V_{\mathrm{bw}})`` on `out` acts as a noisy observation of
`out`, and conditioning the joint normal on it corrects the belief about `in`. The correction
passes through the gain ``K = C (\tilde{V} + V_{\mathrm{bw}})^{-1}``:

```math
m'_{\mathrm{in}} = m_{\mathrm{in}} + K (m_{\mathrm{bw}} - \tilde{m}), \qquad
V'_{\mathrm{in}} = V_{\mathrm{in}} - K C^\top.
```

Only ``\tilde{V} + V_{\mathrm{bw}}`` is inverted, so ``\tilde{V}`` itself may be singular
([A singular forward covariance](@ref)).

For an affine `g` the result is the exact posterior of `in`. For `out = 2in + 1`, with `in`
normal with mean `1` and variance `0.5`, and a backward normal on `out` with mean `4` and
variance `2`:

```@example smoothing
using MessagePassingRulesApproximations

m_in, V_in = 1.0, 0.5
m_tilde, V_tilde, C_tilde = unscented_statistics(Unscented(), x -> 2x + 1, (m_in,), (V_in,))
smoothRTS(m_tilde, V_tilde, C_tilde, m_in, V_in, 4.0, 2.0)
```

The backward normal says `in` is near `(4 - 1) / 2 = 1.5` with variance `2 / 4 = 0.5`. Its
product with the forward belief has mean `1.25` and variance `0.25`, which the smoothed result
reproduces.

## A singular forward covariance

A function with more outputs than inputs gives a singular ``\tilde{V}`` under linearisation: two
outputs of one input vary in one direction only. Take `g(x) = [x, x²]` at a mean of `1`:

```@example smoothing
using LinearAlgebra

g(x) = [x[1], x[1]^2]
m_in, V_in = [1.0], fill(0.1, 1, 1)

A, b = approximate(Linearization(), g, (m_in,))
m_tilde, V_tilde, C_tilde = A * m_in + b, A * V_in * A', V_in * A'
V_tilde, rank(V_tilde)
```

``\tilde{V}`` has rank one, so it has no inverse and no Cholesky factor. A backward message with
a covariance of full rank still makes ``\tilde{V} + V_{\mathrm{bw}}`` invertible, and the
correction runs:

```@example smoothing
m_bw, V_bw = [1.2, 1.5], Matrix(0.2I, 2, 2)
smoothRTS(m_tilde, V_tilde, C_tilde, m_in, V_in, m_bw, V_bw)
```

The linearised relation is affine, `out = A in + b`, so the result is its exact posterior. The
precision form, the prior's precision plus ``A^\top V_{\mathrm{bw}}^{-1} A``, gives the same:

```@example smoothing
precision = inv(V_in) + A' * inv(V_bw) * A
precision \ (inv(V_in) * m_in + A' * inv(V_bw) * (m_bw - b)), inv(precision)
```

## API

```@docs
smoothRTS
```
