"""
    smoothRTS(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out) -> (m_in, V_in)

The Rauch–Tung–Striebel correction of an input's marginal through `out = g(in)`: given the
forward statistics of `g` and a backward message on `out`, the mean and covariance of the
marginal on `in`. It pairs with [`unscented_statistics`](@ref) or a linearisation, which provide
the forward statistics; a Delta node's joint marginal over its inputs is built on it.

# Arguments

- `m_tilde`, `V_tilde`: the mean and covariance of `g(in)` under the forward message on `in`;
- `C_tilde`: the cross-covariance between `in` and `g(in)` under that message;
- `m_fw_in`, `V_fw_in`: the mean and covariance of the forward message on `in`;
- `m_bw_out`, `V_bw_out`: the mean and covariance of the backward message on `out`.

Scalars or vectors and matrices, consistently.

# Returns

`(m_in, V_in)`, the mean and covariance of the marginal on `in`. When `V_tilde` is singular or not
finite, the output carries no uncertainty for the backward message to revise, and the result is
the forward message's `(m_fw_in, V_fw_in)` unchanged.

Petersen, Hoffmann and Rostalski (2018), *On approximate nonlinear Gaussian message passing on
factor graphs*, IEEE Statistical Signal Processing Workshop.
"""
function smoothRTS(
        m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out
    )
    # A singular forward output covariance means the transformed input is deterministic: the
    # node's output carries no uncertainty, so the backward message cannot revise the input and
    # the smoothed inbound marginal is exactly the forward one. This happens whenever an inbound
    # message has zero covariance, in which case `unscented_statistics` (and the linearization
    # equivalent) return `V_tilde = 0` and `C_tilde = 0`.
    #
    # Without this short-circuit, `W_tilde = cholinv(V_tilde)` is `Inf` rather than an error, so
    # `D_tilde = C_tilde * W_tilde` evaluates to `0 * Inf = NaN` and a silently corrupted
    # marginal propagates instead of the correct degenerate one.
    if __rts_is_singular(V_tilde)
        return (m_fw_in, V_fw_in)
    end

    P = cholinv(V_tilde + V_bw_out)
    W_tilde = cholinv(V_tilde)
    D_tilde = C_tilde * W_tilde
    V_in = V_fw_in + D_tilde * (V_bw_out * P * C_tilde' - C_tilde')
    m_out = V_tilde * P * m_bw_out + V_bw_out * P * m_tilde
    m_in = m_fw_in + D_tilde * (m_out - m_tilde)

    return (m_in, V_in) # Statistics for marginal on in
end

__rts_is_singular(V_tilde::Real) = iszero(V_tilde) || !isfinite(V_tilde)
function __rts_is_singular(V_tilde::AbstractMatrix)
    return !all(isfinite, V_tilde) || iszero(det(V_tilde))
end
