@rule(
    form        => Type{ GCV }, 
    on          => :ω,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_y_x::Any, q_z::Any, q_κ::Any),
    meta        => Nothing,
    begin 
        Λ = cov(q_y_x)
        m = mean(q_y_x)

        γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
        γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
        γ_4 = (m[1] - m[2]) ^ 2 + Λ[1, 1] + Λ[2, 2] - Λ[1, 2] - Λ[2, 1]

        a = one(typeof(γ_1))
        b = γ_4 * γ_2
        c = -one(typeof(γ_1))
        d = zero(typeof(γ_1))

        return ExponentialLinearQuadratic(a, b, c, d)
    end
)