export make_node, rule, GCV

struct GCV end

function make_node(::Type{ GCV })
    return FactorNode(GCV, Stochastic, ( :y, :x, :z, :κ, :ω ), ( ( 1, 2 ), ( 3, ), ( 4, ), ( 5, ) ), nothing)
end

function make_node(::Type{ GCV }, y, x, z, κ, ω)
    node = make_node(GCV)
    connect!(node, :y, y)
    connect!(node, :x, x)
    connect!(node, :z, z)
    connect!(node, :κ, κ)
    connect!(node, :ω, ω)
    return node
end

# Message for backward ν_x
@rule(
    form        => Type{ GCV }, 
    on          => :x,
    vconstraint => Marginalisation,
    messages    => (m_y::Any, ),
    marginals   => (q_z::Any, q_κ::Any, q_ω::Any),
    meta        => Nothing,
    begin 
        γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
        γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
        γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))

        return NormalMeanVariance(mean(m_y), var(m_y) + 1.0 / (γ_2 * γ_3))
    end
)

# Message for forward ν_y
@rule(
    form        => Type{ GCV }, 
    on          => :y,
    vconstraint => Marginalisation,
    messages    => (m_x::Any, ),
    marginals   => (q_z::Any, q_κ::Any, q_ω::Any),
    meta        => Nothing,
    begin 
        γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
        γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
        γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))

        return NormalMeanVariance(mean(m_x), var(m_x) + 1.0 / (γ_2 * γ_3))
    end
)

# Message for upward ν_z
@rule(
    form        => Type{ GCV }, 
    on          => :z,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_y_x::Any, q_κ::Any, q_ω::Any),
    meta        => Nothing,
    begin 
        Λ = Matrix(cov(q_y_x))
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

# Message for backward ν_κ
@rule(
    form        => Type{ GCV }, 
    on          => :κ,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_y_x::Any, q_z::Any, q_ω::Any),
    meta        => Nothing,
    begin 
        Λ = Matrix(cov(q_y_x))
        m = mean(q_y_x)

        γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
        γ_4 = (m[1] - m[2]) ^ 2 + Λ[1, 1] + Λ[2, 2] - Λ[1, 2] - Λ[2, 1]

        a = mean(q_z)
        b = γ_4 * γ_3
        c = -a
        d = var(q_z)

        return ExponentialLinearQuadratic(a, b, c, d)
    end
)

# Message for backward ν_ω
@rule(
    form        => Type{ GCV }, 
    on          => :ω,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_y_x::Any, q_z::Any, q_κ::Any),
    meta        => Nothing,
    begin 
        Λ = Matrix(cov(q_y_x))
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


# Marginal for q_xy
@marginalrule(
    form       => Type{ GCV }, 
    on         => :y_x,
    messages   => (m_y::Any, m_x::Any),
    marginals  => (q_z::Any, q_κ::Any, q_ω::Any),
    meta       => Nothing,
    begin 
        γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
        γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
        γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
        γ23 = γ_2 * γ_3

        W = PDMat([ (precision(m_y) + γ23) -γ23; -γ23 (precision(m_x) + γ23) ])
        Λ = inv(W)
        m = Λ * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

        return MvNormalMeanCovariance(m, Λ)
    end
)
