export collectStatistics, concatenateGaussianMV, smoothRTS, marginalizeGaussianMV

# TODO: In RMP there shouldn't be nothing
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

function collectStatistics(msg::NormalDistributionsFamily)
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
