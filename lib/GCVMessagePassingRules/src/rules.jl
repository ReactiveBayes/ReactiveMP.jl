# The GCV rules. Towards `y` and `x`, a normal with the effective noise
# variance 1/⟨e^{-(κz + ω)}⟩; towards `z`, `κ` and `ω`, an ExponentialLinearQuadratic.

# Towards `y` from the message on `x`, or from `q(x)`; towards `x` the mirror image.
noise_variance(q_z, q_κ, q_ω) = exp(-log_noise_precision(q_z, q_κ, q_ω))

function through_noise(m, q_z, q_κ, q_ω)
    m_mean, m_var = mean_var(m)
    return NormalMeanVariance(m_mean, m_var + noise_variance(q_z, q_κ, q_ω))
end

@define_message_update_rule(
    node = GCV, target = :y, args = (m[:x]::UniNormalOrExpLinQuad, q[:z]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> through_noise(args.m[:x], args.q[:z], args.q[:κ], args.q[:ω]),
)

@define_message_update_rule(
    node = GCV, target = :y, args = (q[:x]::Any, q[:z]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> NormalMeanVariance(mean(args.q[:x]), noise_variance(args.q[:z], args.q[:κ], args.q[:ω])),
)

@define_message_update_rule(
    node = GCV, target = :x, args = (m[:y]::UniNormalOrExpLinQuad, q[:z]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> through_noise(args.m[:y], args.q[:z], args.q[:κ], args.q[:ω]),
)

@define_message_update_rule(
    node = GCV, target = :x, args = (q[:y]::Any, q[:z]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> NormalMeanVariance(mean(args.q[:y]), noise_variance(args.q[:z], args.q[:κ], args.q[:ω])),
)

# Towards `z`: exp(-(⟨κ⟩ z + ⟨(y - x)²⟩ ⟨e^{-ω}⟩ exp(-⟨κ⟩ z + Var(κ) z² / 2)) / 2).
function towards_z(method, psi, q_κ, q_ω)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)
    return ExponentialLinearQuadratic(method, κ_mean, psi * exp(-ω_mean + ω_var / 2), -κ_mean, κ_var)
end

@define_message_update_rule(
    node = GCV, target = :z, args = (q[:y, :x]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (algo, args) -> towards_z(algo.method, expected_square_difference(args.q[:y, :x]), args.q[:κ], args.q[:ω]),
)

@define_message_update_rule(
    node = GCV, target = :z, args = (q[:y]::Any, q[:x]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (algo, args) -> towards_z(algo.method, expected_square_difference(args.q[:y], args.q[:x]), args.q[:κ], args.q[:ω]),
)

# Towards `κ`: the same form with the roles of κ and z exchanged.
function towards_κ(method, psi, q_z, q_ω)
    z_mean, z_var = mean_var(q_z)
    ω_mean, ω_var = mean_var(q_ω)
    return ExponentialLinearQuadratic(method, z_mean, psi * exp(-ω_mean + ω_var / 2), -z_mean, z_var)
end

@define_message_update_rule(
    node = GCV, target = :κ, args = (q[:y, :x]::Any, q[:z]::Any, q[:ω]::Any),
    body = (algo, args) -> towards_κ(algo.method, expected_square_difference(args.q[:y, :x]), args.q[:z], args.q[:ω]),
)

@define_message_update_rule(
    node = GCV, target = :κ, args = (q[:y]::Any, q[:x]::Any, q[:z]::Any, q[:ω]::Any),
    body = (algo, args) -> towards_κ(algo.method, expected_square_difference(args.q[:y], args.q[:x]), args.q[:z], args.q[:ω]),
)

# Towards `ω`: exp(-(ω + ⟨(y - x)²⟩ ⟨e^{-κz}⟩ e^{-ω}) / 2), with ⟨e^{-κz}⟩ as in log_noise_precision.
function towards_ω(method, psi, q_z, q_κ)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    γ = z_mean^2 * κ_var + κ_mean^2 * z_var + z_var * κ_var
    T = typeof(γ)
    return ExponentialLinearQuadratic(method, one(T), psi * exp(-κ_mean * z_mean + γ / 2), -one(T), zero(T))
end

@define_message_update_rule(
    node = GCV, target = :ω, args = (q[:y, :x]::Any, q[:z]::Any, q[:κ]::Any),
    body = (algo, args) -> towards_ω(algo.method, expected_square_difference(args.q[:y, :x]), args.q[:z], args.q[:κ]),
)

@define_message_update_rule(
    node = GCV, target = :ω, args = (q[:y]::Any, q[:x]::Any, q[:z]::Any, q[:κ]::Any),
    body = (algo, args) -> towards_ω(algo.method, expected_square_difference(args.q[:y], args.q[:x]), args.q[:z], args.q[:κ]),
)

# The joint of y and x: their messages coupled by the effective noise precision.
@define_marginal_update_rule(
    node = GCV, target = (:y, :x), args = (m[:y]::UniNormalOrExpLinQuad, m[:x]::UniNormalOrExpLinQuad, q[:z]::Any, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> begin
        y_mean, y_precision = mean_precision(args.m[:y])
        x_mean, x_precision = mean_precision(args.m[:x])
        ab = exp(log_noise_precision(args.q[:z], args.q[:κ], args.q[:ω]))
        MvNormalWeightedMeanPrecision([y_mean * y_precision; x_mean * x_precision], [y_precision + ab -ab; -ab x_precision + ab])
    end,
)
