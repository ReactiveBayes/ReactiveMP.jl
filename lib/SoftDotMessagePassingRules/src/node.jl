"""
    SoftDot

The SoftDot node, the density

    p(y | θ, x, γ) = N(y | θᵀx, γ⁻¹),

a dot product of `θ` and `x` softened by Gaussian noise of precision `γ`. A stochastic node
with the interfaces, in order:

1. `y`, the result, a scalar;
2. `θ` (alias `theta`), the first factor, a scalar or a vector;
3. `x`, the second factor, of the same dimension as `θ`;
4. `γ` (alias `gamma`), the precision of the noise, a positive scalar.

It runs under [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which a model
need not name. Its rules are variational and in closed form: every message reads the marginals
of the other interfaces, under the mean-field `q(y)q(θ)q(x)q(γ)` or the structured
`q(y, x)q(θ)q(γ)`, where the rules towards `y` and `x` read the message on the other one and the
joint `q(y, x)` has its own rule. The marginals on `y`, `θ` and `x` are normals and the one on
`γ` a gamma: the rules read their means and covariances, and the mean and the mean of the
logarithm of `q(γ)`.

There are no belief-propagation rules, and other factorisations, such as a joint over `θ` and
`x`, have no rules. There is an average energy for the two factorisations above.

See also: [`softdot`](@ref)
"""
struct SoftDot end

"""
    softdot

An alias of [`SoftDot`](@ref), the node `y ~ N(θᵀx, γ⁻¹)`.
"""
const softdot = SoftDot

@define_factor_node(node = SoftDot, type = Stochastic, interfaces = [:y, (:θ, aliases = [:theta]), :x, (:γ, aliases = [:gamma])])

# ⟨-log N(y | θᵀx, γ⁻¹)⟩ = ½(log 2π - ⟨log γ⟩ + ⟨γ⟩ ⟨(y - θᵀx)²⟩), with the whole bracket under a
# single ⟨γ⟩, the cross term included.
softdot_energy(q_γ, expected_square) = (-mean(log, q_γ) + log2π + mean(q_γ) * expected_square) / 2

@define_average_energy(
    node = SoftDot, args = (q[:y]::Any, q[:θ]::Any, q[:x]::Any, q[:γ]::Any),
    body = (args) -> begin
        m_y, V_y = mean_cov(args.q[:y])
        m_θ, V_θ = mean_cov(args.q[:θ])
        m_x, V_x = mean_cov(args.q[:x])
        expected_square = V_y + m_y^2 - 2 * m_y * dot(m_θ, m_x) + StandardMessagePassingRules.mul_trace(V_θ, V_x) + dot(m_x, V_θ, m_x) + dot(m_θ, V_x + m_x * m_x', m_θ)
        softdot_energy(args.q[:γ], expected_square)
    end,
)

@define_average_energy(
    node = SoftDot, args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:θ]::Any, q[:γ]::Any),
    body = (args) -> begin
        m_θ, V_θ = mean_cov(args.q[:θ])
        m_y, V_y, m_x, V_x, V_xy = split_y_x(args.q[:y, :x])
        expected_square = V_y + m_y^2 - 2 * dot(m_θ, V_xy + m_x * m_y) + StandardMessagePassingRules.mul_trace(V_θ, V_x) + dot(m_x, V_θ, m_x) + dot(m_θ, V_x, m_θ) + abs2(dot(m_θ, m_x))
        softdot_energy(args.q[:γ], expected_square)
    end,
)
