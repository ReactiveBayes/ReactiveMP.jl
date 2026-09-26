# Smoothing

A deterministic node `out = g(in)` sends forward the moments of `g(in)` under the message on
`in`. When a message comes back on `out`, the marginal on `in` is corrected by a
Rauch–Tung–Striebel step, from the forward statistics of either transform: [`smoothRTS`](@ref).

```@docs
smoothRTS
```
