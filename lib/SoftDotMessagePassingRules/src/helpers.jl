# What v6's SoftDot took from the autoregressive node, written out so that this package does not
# depend on AR's: the slicing of a joint q(y, x) (v6's `ar_slice`) and AR's `y` message from a
# message on `x`, of which SoftDot kept the first component.

# The blocks of a joint q(y, x) over a scalar `y` and an `x` of length `order`: the means and
# variances of `y` and of `x`, and the cross-covariance Cov(x, y). They are scalars when `x` is
# (order 1), as v6's `ar_slice` gave them.
function split_y_x(q_y_x)
    m, V = mean_cov(q_y_x)
    order = length(m) - 1
    if order == 1
        return m[1], V[1, 1], m[2], V[2, 2], V[2, 1]
    else
        x = 2:(order + 1)
        return m[1], V[1, 1], m[x], V[x, x], V[x, 1]
    end
end

# AR's `y` message from the message on `x`, reduced to its first component. The AR(order)
# prediction is A x plus noise of variance 1/⟨γ⟩ on the first component, A the companion matrix of
# ⟨θ⟩; with D = W_x + ⟨γ⟩ V_θ its mean is A D⁻¹ W_x m_x and its covariance A D⁻¹ Aᵀ + noise. The
# first component reads only A's first row, ⟨θ⟩ᵀ, so the companion matrix is never formed:
#
#     mean = ⟨θ⟩ᵀ D⁻¹ W_x m_x,    variance = ⟨θ⟩ᵀ D⁻¹ ⟨θ⟩ + 1/⟨γ⟩
#
# (D is symmetric, so D⁻¹⟨θ⟩ serves for both).
function y_from_x(m_x, q_θ, q_γ)
    mθ, Vθ = mean_cov(q_θ)
    mx, Wx = mean_invcov(m_x)
    mγ = mean(q_γ)
    c = (Wx + mγ * Vθ) \ mθ
    return NormalMeanVariance(dot(c, Wx * mx), dot(c, mθ) + inv(mγ))
end
