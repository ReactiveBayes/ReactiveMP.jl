# `out = dot(in1, in2)`: the function `dot` is the node. One input must be known; two
# Gaussian inputs have no closed form, and SoftDot is the node for them.
@define_factor_node(node = dot, type = Deterministic, interfaces = [:out, :in1, :in2])

# The default: a zero on the precision's diagonal is replaced, so it stays invertible.
dot_default_correction() = ReplaceZeroDiagonalEntries(tiny)

const DOT_OF_NORMALS = "The rule for the dot product node between two NormalDistributionsFamily instances is not available in closed form. Please use SoftDot instead."

# out = aᵀx for a known a and a Gaussian x: N(aᵀE[x], aᵀ Cov[x] a).
dot_forward(a, m_x) = ((μ, Σ) = mean_cov(m_x); NormalMeanVariance(dot(a, μ), dot(a, Σ, a)))

# The likelihood of x from out = aᵀx: weighted mean a ξ_out and precision a w_out aᵀ, rank one
# for a vector a, so it goes through the matrix correction.
function dot_backward(ctx, m_a::PointMass, m_out)
    a = mean(m_a)
    ξ, w = weightedmean_precision(m_out)
    W = correction!(matrix_correction(ctx, dot_default_correction()), v_a_vT(a, w))
    return promote_variate_type(variate_form(typeof(m_a)), NormalWeightedMeanPrecision)(a * ξ, W)
end
