using ForwardDiff

"""
Return local linearization of g around expansion point x_hat
for Delta node with single input interface
"""
function localLinearizationSingleIn(g::Function, x_hat::Float64)
    a = ForwardDiff.derivative(g, x_hat)
    b = g(x_hat) - a * x_hat

    return (a, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with multiple input interfaces
"""
function localLinearizationMultiIn(g::Function, x_hat::Vector{Float64})
    g_unpacked(x::Vector) = g(x...)
    A = ForwardDiff.gradient(g_unpacked, x_hat)'
    b = g(x_hat...) - A * x_hat

    return (A, b)
end

function collectStatistics(msgs::NTuple{N, Any}) where {N}
    stats = []
    for msg in msgs
        (msg === nothing) && continue # Skip unreported messages
        push!(stats, mean_cov(msg.dist))
    end

    ms = [stat[1] for stat in stats]
    Vs = [stat[2] for stat in stats]

    return (ms, Vs) # Return tuple with vectors for means and covariances
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

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{N, Any}, meta::DeltaExtended{T}) where {f, N, T} =
    begin
        (ms_fw_in, Vs_fw_in) = collectStatistics(m_ins...) # Returns arrays with individual means and covariances
        (A, b) = localLinearizationMultiIn(meta.inverse, ms_fw_in)
        (m_fw_in, V_fw_in, _) = concatenateGaussianMV(ms_fw_in, Vs_fw_in)
        m = A * m_fw_in + b
        V = A * V_fw_in * A'

        F = size(m, 1) == 1 ? Univariate : Multivariate
        return convert(promote_variate_type(F, NormalMeanVariance), m, V)
    end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_in::NTuple{1, Any}, meta::DeltaExtended{T}) where {f, T} = begin
    μ_in, Σ_in = mean_cov(m_in)
    (A, b) = localLinearizationSingleIn(f, m_in)
    m = A * μ_in + b
    V = A * Σ_in * A'
    F = size(m, 1) == 1 ? Univariate : Multivariate
    return convert(promote_variate_type(F, NormalMeanVariance), m, V)
end
