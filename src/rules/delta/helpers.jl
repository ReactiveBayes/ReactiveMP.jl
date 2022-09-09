import Base: split

export localLinearizationSingleIn,
    localLinearizationMultiIn, collectStatistics, concatenateGaussianMV, smoothRTS, marginalizeGaussianMV,
    unscentedStatistics, sigmaPointsAndWeights

using ForwardDiff

"""
Return local linearization of g around expansion point x_hat
for Delta node with single input interface
"""
function localLinearizationSingleIn(g::Any, x_hat::Float64)
    a = ForwardDiff.derivative(g, x_hat)
    b = g(x_hat) - a * x_hat

    return (a, b)
end

function localLinearizationSingleIn(g::Any, x_hat::Vector{Float64})
    A = ForwardDiff.jacobian(g, x_hat)
    @show "here"
    b = g(x_hat) - A * x_hat

    return (A, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with multiple input interfaces
"""
function localLinearizationMultiIn(g::Any, x_hat::Vector)
    g_unpacked(x::Vector) = g(x...)
    @show x_hat
    @show typeof(x_hat)
    A = ForwardDiff.gradient(g_unpacked, x_hat)'
    b = g(x_hat...) - A * x_hat

    return (A, b)
end

"""
Concatenate a vector (of vectors and floats) and return with original dimensions (for splitting)
"""
function concatenate(xs::Vector)
    ds = size.(xs) # Extract dimensions
    x = vcat(vec.(xs)...)

    return (x, ds)
end

"""
Return integer dimensionality
"""
intdim(tup::Tuple) = prod(tup) # Returns 1 for ()

"""
Split a vector in chunks of lengths specified by ds.
"""
function split(vec::Vector, ds::Vector{<:Tuple})
    N = length(ds)
    res = Vector{Any}(undef, N)

    d_start = 1
    for k in 1:N # For each original statistic
        d_end = d_start + intdim(ds[k]) - 1

        if ds[k] == () # Univariate
            res[k] = vec[d_start] # Return scalar
        else # Multi- of matrix variate
            res[k] = reshape(vec[d_start:d_end], ds[k]) # Return vector or matrix
        end

        d_start = d_end + 1
    end

    return res
end

function localLinearizationMultiIn(g::Any, x_hat::Vector{Vector{Float64}})
    (x_cat, ds) = concatenate(x_hat)
    g_unpacked(x::Vector) = g(split(x, ds)...)
    A = ForwardDiff.jacobian(g_unpacked, x_cat)
    b = g(x_hat...) - A * x_cat
    return (A, b)
end

function collectStatistics(msgs::Vararg{Union{Any, Nothing}})
    stats = []
    for msg in msgs
        (msg === nothing) && continue # Skip unreported messages
        push!(stats, mean_cov(msg))
    end

    ms = [stat[1] for stat in stats]
    Vs = [stat[2] for stat in stats]
    return (ms, Vs) # Return tuple with vectors for means and covariances
end

function collectStatistics(msg::Any)
    return mean_cov(msg)
end

"""
Concatenate independent means and (co)variances of separate Gaussians in a unified mean and covariance.
Additionally returns a vector with the original dimensionalities, so statistics can later be re-separated.
"""
function concatenateGaussianMV(ms::Vector, Vs::Vector)
    # Extract dimensions
    ds = [size(m_k) for m_k in ms]
    dl = intdim.(ds)
    d_in_tot = sum(dl)

    # Initialize concatenated statistics
    m = zeros(d_in_tot)
    V = zeros(d_in_tot, d_in_tot)

    # Construct concatenated statistics
    d_start = 1
    for k in 1:length(ms) # For each inbound statistic
        d_end = d_start + dl[k] - 1
        if ds[k] == () # Univariate
            m[d_start] = ms[k]
            V[d_start, d_start] = Vs[k]
        else # Multivariate
            m[d_start:d_end] = ms[k]
            V[d_start:d_end, d_start:d_end] = Vs[k]
        end
        d_start = d_end + 1
    end

    return (m, V, ds) # Return concatenated mean and covariance with original dimensions (for splitting)
end

"""
RTS smoother update for backward message
"""
function smoothRTSMessage(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)
    C_tilde_inv = pinv(C_tilde)
    V_bw_in = V_fw_in * C_tilde_inv' * (V_tilde + V_bw_out) * C_tilde_inv * V_fw_in - V_fw_in
    m_bw_in = m_fw_in + V_fw_in * C_tilde_inv' * (m_bw_out - m_tilde)

    return (m_bw_in, V_bw_in) # Statistics for backward message on in
end

"""
RTS smoother update for inbound marginal; based on (Petersen et al. 2018; On Approximate Delta Gaussian Message Passing on Factor Graphs)
"""
function smoothRTS(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)
    P = cholinv(V_tilde + V_bw_out)
    W_tilde = cholinv(V_tilde)
    D_tilde = C_tilde * W_tilde
    V_in = V_fw_in + D_tilde * (V_bw_out * P * C_tilde' - C_tilde')
    m_out = V_tilde * P * m_bw_out + V_bw_out * P * m_tilde
    m_in = m_fw_in + D_tilde * (m_out - m_tilde)

    return (m_in, V_in) # Statistics for marginal on in
end

"""
Return the marginalized statistics of the Gaussian corresponding to an inbound inx
"""
function marginalizeGaussianMV(m::Vector{Float64}, V::AbstractMatrix, ds::Vector, inx::Int64)
    if ds[inx] == () # Univariate original
        return (m[inx], V[inx, inx]) # Return scalars
    else # Multivariate original
        dl = intdim.(ds)
        dl_start = cumsum([1; dl]) # Starting indices
        d_start = dl_start[inx]
        d_end = dl_start[inx+1] - 1
        mx = m[d_start:d_end] # Vector
        Vx = V[d_start:d_end, d_start:d_end] # Matrix
        return (mx, Vx)
    end
end

# Unscented transform

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
