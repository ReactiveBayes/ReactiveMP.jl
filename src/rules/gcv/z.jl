@rule(
    form        => Type{ GCV }, 
    on          => :z,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_y_x::Any, q_κ::Any, q_ω::Any),
    meta        => Nothing,
    begin 
        Λ = cov(q_y_x)
        m = mean(q_y_x)

        γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
        γ_4 = (m[1] - m[2]) ^ 2 + Λ[1, 1] + Λ[2, 2] - Λ[1, 2] - Λ[2, 1]
        γ_5 = γ_4 * γ_3 * exp(-mean(q_κ))

        a = mean(q_κ)
        b = γ_4 * γ_3
        c = -a
        d = var(q_κ)

        return ExponentialLinearQuadratic(a, b, c, d)
    end
)