"""
    RuleContext(; node = nothing, product = nothing, linalg = nothing, rng = nothing)

The read-only infrastructure a rule receives as `ctx`. It never takes part in dispatch.

- `node`: the factor node the rule belongs to, e.g. for `getnodefn(ctx.node, …)`.
- `product`: `(left, right) -> (distribution, logscale)`.
- `linalg`: the linear-algebra strategy. **Unstable**: its protocol is not settled yet.
- `rng`: the random number generator, owned by the caller.

A service a rule needs and does not get is `nothing`; see [`missing_services`](@ref).
"""
struct RuleContext{N, P, L, R}
    node::N
    product::P
    linalg::L
    rng::R
end

RuleContext(; node = nothing, product = nothing, linalg = nothing, rng = nothing) =
    RuleContext(node, product, linalg, rng)

const CONTEXT_SERVICES = (:node, :product, :linalg, :rng)
