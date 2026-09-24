"""
    SoftDot

The node `y ~ N(dot(θ, x), γ⁻¹)`, a substitute for the dot product node that softens its delta
constraint with Gaussian noise of precision `γ`. Its interfaces:

1. `y`, the result of the soft dot product,
2. `θ` (alias `theta`), the first factor,
3. `x`, the second factor,
4. `γ` (alias `gamma`), the precision of the noise.

Its rules are variational, in closed form, under mean-field or a structured `q(y, x)`; there
are no belief-propagation rules. It runs under `DefaultAlgorithm`.

See also: [`softdot`](@ref)
"""
struct SoftDot end

"""
    softdot

An alias of [`SoftDot`](@ref).
"""
const softdot = SoftDot

@define_factor_node(node = SoftDot, type = Stochastic, interfaces = [:y, (:θ, aliases = [:theta]), :x, (:γ, aliases = [:gamma])])

# ⟨-log N(y | θᵀx, γ⁻¹)⟩ = ½(log 2π - ⟨log γ⟩ + ⟨γ⟩ ⟨(y - θᵀx)²⟩), with the whole bracket under a
# single ⟨γ⟩ (ReactiveMP.jl#615: v6 once had a second ⟨γ⟩ on the cross term).
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
