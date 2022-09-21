export unscentedStatistics, sigmaPointsAndWeights

"""
Return the statistics for the unscented approximation to the forward joint
"""
# Single univariate inbound
function unscentedStatistics(
    m::Float64,
    V::Float64,
    g::Any;
    alpha = default_alpha,
    beta = default_beta,
    kappa = default_kappa
)
    (sigma_points, weights_m, weights_c) = sigmaPointsAndWeights(m, V; alpha = alpha, beta = beta, kappa = kappa)

    g_sigma = g.(sigma_points)
    m_tilde = sum(weights_m .* g_sigma)
    V_tilde = sum(weights_c .* (g_sigma .- m_tilde) .^ 2)
    C_tilde = sum(weights_c .* (sigma_points .- m) .* (g_sigma .- m_tilde))

    return (m_tilde, V_tilde, C_tilde)
end

# Single multivariate inbound
function unscentedStatistics(
    m::Vector{Float64},
    V::AbstractMatrix,
    g::Any;
    alpha = default_alpha,
    beta = default_beta,
    kappa = default_kappa
)
    (sigma_points, weights_m, weights_c) = sigmaPointsAndWeights(m, V; alpha = alpha, beta = beta, kappa = kappa)
    d = length(m)

    g_sigma = g.(sigma_points)
    m_tilde = sum([weights_m[k+1] * g_sigma[k+1] for k in 0:2*d])
    V_tilde = sum([weights_c[k+1] * (g_sigma[k+1] - m_tilde) * (g_sigma[k+1] - m_tilde)' for k in 0:2*d])
    C_tilde = sum([weights_c[k+1] * (sigma_points[k+1] - m) * (g_sigma[k+1] - m_tilde)' for k in 0:2*d])

    return (m_tilde, V_tilde, C_tilde)
end

# Multiple inbounds of possibly mixed variate type
function unscentedStatistics(
    ms::Vector,
    Vs::Vector,
    g::Any;
    alpha = default_alpha,
    beta = default_beta,
    kappa = default_kappa
)
    (m, V, ds) = concatenateGaussianMV(ms, Vs)
    (sigma_points, weights_m, weights_c) = sigmaPointsAndWeights(m, V; alpha = alpha, beta = beta, kappa = kappa)

    g_sigma = [g(split(sp, ds)...) for sp in sigma_points] # Unpack each sigma point in g

    d = sum(intdim.(ds)) # Dimensionality of joint
    m_tilde = sum([weights_m[k+1] * g_sigma[k+1] for k in 0:2*d]) # Vector
    V_tilde = sum([weights_c[k+1] * (g_sigma[k+1] - m_tilde) * (g_sigma[k+1] - m_tilde)' for k in 0:2*d]) # Matrix
    C_tilde = sum([weights_c[k+1] * (sigma_points[k+1] - m) * (g_sigma[k+1] - m_tilde)' for k in 0:2*d]) # Matrix

    return (m_tilde, V_tilde, C_tilde)
end

"""
Return the sigma points and weights for a Gaussian distribution
"""
function sigmaPointsAndWeights(
    m::Float64,
    V::Float64;
    alpha = default_alpha,
    beta = default_beta,
    kappa = default_kappa
)
    lambda = (1 + kappa) * alpha^2 - 1

    sigma_points = Vector{Float64}(undef, 3)
    weights_m = Vector{Float64}(undef, 3)
    weights_c = Vector{Float64}(undef, 3)

    l = sqrt((1 + lambda) * V)

    sigma_points[1] = m
    sigma_points[2] = m + l
    sigma_points[3] = m - l
    weights_m[1] = lambda / (1 + lambda)
    weights_m[2] = weights_m[3] = 1 / (2 * (1 + lambda))
    weights_c[1] = weights_m[1] + (1 - alpha^2 + beta)
    weights_c[2] = weights_c[3] = 1 / (2 * (1 + lambda))

    return (sigma_points, weights_m, weights_c)
end

function sigmaPointsAndWeights(
    m::Vector{Float64},
    V::AbstractMatrix;
    alpha = default_alpha,
    beta = default_beta,
    kappa = default_kappa
)
    d = length(m)
    lambda = (d + kappa) * alpha^2 - d

    sigma_points = Vector{Vector{Float64}}(undef, 2 * d + 1)
    weights_m = Vector{Float64}(undef, 2 * d + 1)
    weights_c = Vector{Float64}(undef, 2 * d + 1)

    if isa(V, Diagonal)
        L = sqrt((d + lambda) * V) # Matrix square root
    else
        L = sqrt(Hermitian((d + lambda) * V))
    end

    sigma_points[1] = m
    weights_m[1] = lambda / (d + lambda)
    weights_c[1] = weights_m[1] + (1 - alpha^2 + beta)
    for i in 1:d
        sigma_points[2*i] = m + L[:, i]
        sigma_points[2*i+1] = m - L[:, i]
    end
    weights_m[2:end] .= 1 / (2 * (d + lambda))
    weights_c[2:end] .= 1 / (2 * (d + lambda))

    return (sigma_points, weights_m, weights_c)
end
