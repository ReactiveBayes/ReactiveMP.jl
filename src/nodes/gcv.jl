export GCVNode

struct GCV end

function GCVNode()
    return FactorNode(GCV, ( :x, :y, :z, :κ, :ω ), ( ( 1, 2 ), ( 3, ), ( 4, ), ( 5, ) ))
end

# Message for backward ν_x
function rule(::Type{ <: GCV }, ::Val{:x}, ::Marginalisation, messages::Tuple{Message}, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    m_y = messages[1]
    q_z = marginals[1]
    q_κ = marginals[2]
    q_ω = marginals[3]

    γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
    γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))

    return NormalMeanVariance(mean(m_y), var(m_y) + γ_2 * γ_3)
end

# Message for forward ν_y
function rule(::Type{ <: GCV }, ::Val{:y}, ::Marginalisation, messages::Tuple{Message}, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)

    m_x = messages[1]
    q_z = marginals[1]
    q_κ = marginals[2]
    q_ω = marginals[3]

    γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
    γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))

    return NormalMeanVariance(mean(m_x), var(m_x) + γ_2 * γ_3)
end

# Message for upward ν_z
function rule(::Type{ <: GCV }, ::Val{:z}, ::Marginalisation, messages::Nothing, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    q_xy = marginals[1]
    q_κ  = marginals[2]
    q_ω  = marginals[3]

    Λ = Matrix(cov(q_xy))
    m = mean(q_xy)

    γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
    γ_4 = (m[1] - m[2]) ^ 2 + Λ[1, 1] + Λ[2, 2] - Λ[1, 2] - Λ[2, 1]
    γ_5 = γ_4 * γ_3 * exp(-mean(q_κ))

    a = mean(q_κ)
    b = γ_4 * γ_3
    c = -a
    d = var(q_κ)

    return ExponentialLinearQuadratic(a, b, c, d)
end

# Message for backward ν_κ
function rule(::Type{ <: GCV }, ::Val{:κ}, ::Marginalisation, messages::Nothing, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    q_xy = marginals[1]
    q_z  = marginals[2]
    q_ω  = marginals[3]

    Λ = Matrix(cov(q_xy))
    m = mean(q_xy)

    γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
    γ_4 = (m[1] - m[2]) ^ 2 + Λ[1, 1] + Λ[2, 2] - Λ[1, 2] - Λ[2, 1]

    a = mean(q_z)
    b = γ_4 * γ_3
    c = -a
    d = var(q_z)

    return ExponentialLinearQuadratic(a, b, c, d)
end

# Message for backward ν_ω
function rule(::Type{ <: GCV }, ::Val{:ω}, ::Marginalisation, messages::Nothing, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    q_xy = marginals[1]
    q_z  = marginals[2]
    q_κ  = marginals[3]

    Λ = Matrix(cov(q_xy))
    m = mean(q_xy)

    γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
    γ_4 = (m[1] - m[2]) ^ 2 + Λ[1, 1] + Λ[2, 2] - Λ[1, 2] - Λ[2, 1]

    a = one(typeof(γ_1))
    b = γ_4 * γ_2
    c = -one(typeof(γ_1))
    d = zero(typeof(γ_1))

    return ExponentialLinearQuadratic(a, b, c, d)
end

# Marginal for q_xy
function marginalrule(::Type{ <: GCV }, ::Val{:xy}, messages::Tuple{Message, Message}, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    m_x = messages[1]
    m_y = messages[2]

    q_z = marginals[1]
    q_κ = marginals[2]
    q_ω = marginals[3]

    γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
    γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
    γ23 = γ_2 * γ_3

    W = PDMat([ (precision(m_x) + γ23) -γ23; -γ23 (precision(m_y) + γ23) ])
    Λ = inv(W)
    m = Λ * [ mean(m_x) * precision(m_x); mean(m_y) * precision(m_y) ]

    return MvNormalMeanCovariance(m, Λ)
end
