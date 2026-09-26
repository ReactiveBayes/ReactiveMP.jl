"""
    HalfNormal

The half-normal node: `out` is the absolute value of a zero-mean normal with variance `v`,

```math
p(\\mathrm{out} \\mid v) = \\frac{2}{\\sqrt{2π v}} \\exp\\left(-\\frac{\\mathrm{out}^2}{2v}\\right),
\\quad \\mathrm{out} \\ge 0.
```

Its interfaces are `out` and `v`, with the aliases `var` and `σ²`; it runs under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm).

**Rules.** Only the message towards `out`, from a known variance (a `PointMass` marginal of `v`):
a `Normal(0, √v)` truncated to `[0, ∞)`. There is **no rule towards `v`**, so `v` must be
known, and no marginal rule. The average energy takes any marginals whose `mean_var` and
`mean(log, ·)`, `mean(inv, ·)` exist.

# Examples

The average energy of a known `out = 1` under a known variance `v = 1`, `-log p(1 | 1)`:

```jldoctest; setup = :(using StandardMessagePassingRules, MessagePassingRulesBase, BayesBase)
julia> energy = getresult(@call_average_energy(node = HalfNormal, q = (out = PointMass(1.0), v = PointMass(1.0))));

julia> energy ≈ (log(π / 2) + 1) / 2
true
```
"""
struct HalfNormal end

@define_factor_node(node = HalfNormal, type = Stochastic, interfaces = [:out, (:v, aliases = [:var, :σ²])])

@define_average_energy(
    node = HalfNormal,
    args = (q[:out]::Any, q[:v]::Any),
    body = (args) -> begin
        out_mean, out_var = mean_var(args.q[:out])
        (log(π / 2) + mean(log, args.q[:v]) + mean(inv, args.q[:v]) * (out_mean^2 + out_var)) / 2
    end,
)
