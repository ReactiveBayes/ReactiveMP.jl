"""
    RuleContext(; node = nothing, product = nothing, linalg = nothing, rng = nothing, matrix_correction = nothing)

The read-only infrastructure a rule receives as `ctx`. It never takes part in dispatch.

- `node`: the factor node the rule belongs to, e.g. for [`getnodefn`](@ref)`(ctx.node, target)`.
- `product`: `(left, right) -> (distribution, logscale)`.
- `linalg`: the linear-algebra strategy. **Unstable**: its protocol is not settled yet.
- `rng`: the random number generator, owned by the caller.
- `matrix_correction`: how a rule corrects a matrix it builds before using it, such as a
  precision that must stay positive definite: a strategy from MatrixCorrectionTools, applied
  as `correction!(ctx.matrix_correction, M)`. `nothing` applies no correction.

A service a rule needs and does not get is `nothing`; see [`missing_services`](@ref). For an
optional service, `matrix_correction`, `nothing` is a setting rather than an absence.
"""
struct RuleContext{N, P, L, R, M}
    node::N
    product::P
    linalg::L
    rng::R
    matrix_correction::M
end

RuleContext(; node = nothing, product = nothing, linalg = nothing, rng = nothing, matrix_correction = nothing) =
    RuleContext(node, product, linalg, rng, matrix_correction)

const CONTEXT_SERVICES = (:node, :product, :linalg, :rng, :matrix_correction)

# The services for which `nothing` is a setting, so that a rule declaring one is never short of it.
const OPTIONAL_CONTEXT_SERVICES = (:matrix_correction,)
