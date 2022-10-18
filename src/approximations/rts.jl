
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