export localLinearizationSingleIn, localLinearizationMultiIn, collectStatistics, concatenateGaussianMV, smoothRTS

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
    b = g(x_hat) - A * x_hat

    return (A, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with multiple input interfaces
"""
function localLinearizationMultiIn(g::Any, x_hat::Vector{Float64})
    g_unpacked(x::Vector) = g(x...)
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
    g_unpacked(x_cat)
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
