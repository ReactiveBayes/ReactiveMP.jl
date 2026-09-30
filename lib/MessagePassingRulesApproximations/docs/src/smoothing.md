# Smoothing

A deterministic relation `out = g(in)` sends forward the moments of `g(in)` under what is known
about `in`. When information about `out` arrives from the other direction, the belief about `in`
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

A backward normal ``N(m_{\mathrm{bw}}, V_{\mathrm{bw}})`` on `out` multiplies the forward one,
which gives the corrected belief ``N(m_{\mathrm{out}}, V_{\mathrm{out}})`` about `out`. The
correction then passes to `in` through the gain ``D = C \tilde{V}^{-1}``:

```math
m'_{\mathrm{in}} = m_{\mathrm{in}} + D (m_{\mathrm{out}} - \tilde{m}), \qquad
V'_{\mathrm{in}} = V_{\mathrm{in}} + D (V_{\mathrm{out}} - \tilde{V}) D^\top.
```

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

## API

```@docs
smoothRTS
```
