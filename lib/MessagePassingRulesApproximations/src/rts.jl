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

`(m_in, V_in)`, the mean and covariance of the marginal on `in`. Only `V_tilde + V_bw_out` is
inverted, so `V_tilde` may be singular: a linearisation of a `g` with more outputs than inputs makes
it so. When `V_tilde` is not finite, the result is the forward message's `(m_fw_in, V_fw_in)`
unchanged.

Petersen, Hoffmann and Rostalski (2018), *On approximate nonlinear Gaussian message passing on
factor graphs*, IEEE Statistical Signal Processing Workshop.
"""
function smoothRTS(
        m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out
    )
    # A forward output covariance that is not finite leaves nothing to correct with.
    all(isfinite, V_tilde) || return (m_fw_in, V_fw_in)
    # The backward message conditions the joint normal of `in` and `out`, a Kalman update that
    # inverts only `V_tilde + V_bw_out`. `V_tilde` itself may be singular, as a linearisation
    # from fewer inputs than outputs makes it, and a zero cross-covariance, as a zero-covariance
    # input gives, returns the forward statistics.
    K = C_tilde * cholinv(V_tilde + V_bw_out)
    return (m_fw_in + K * (m_bw_out - m_tilde), V_fw_in - K * C_tilde')
end
