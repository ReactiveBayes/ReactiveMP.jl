# Whether `vec` has exactly one entry approximately equal to one and all others approximately
# zero, to within `sqrt(eps)` of its element type.
function isonehot(vec::AbstractVector{T}) where {T}
    ones_seen = 0
    atol = sqrt(eps(T))
    for e in vec
        if isapprox(e, one(e); atol)
            ones_seen += 1
        elseif !isapprox(e, zero(e); atol)
            return false
        end
    end
    return ones_seen == 1
end

# `cluster` with every block in the float type of all `inputs` together. A rule's output must
# carry the promoted float type of every input, and that includes a block that passes an input
# through unchanged, such as v6's `v = m_v`.
promoted_cluster(cluster::FactorizedCluster, inputs...) =
    BayesBase.convert_paramfloattype(BayesBase.promote_paramfloattype(inputs...), cluster)

"""
    diageye(::Type{T}, n)

The `n`×`n` identity matrix of element type `T`, dense.
"""
diageye(::Type{T}, n::Integer) where {T} = Matrix{T}(I, n, n)

# The precision of the joint of `out` and `μ` when `out` is normal around `μ` with precision
# `W_bar` and each carries a message of precision `W_out`, `W_μ`:
# `[W_out + W_bar  -W_bar; -W_bar  W_μ + W_bar]`, for scalars and matrices alike.
coupled_precision(W_out, W_μ, W_bar) = [W_out + W_bar -W_bar; -W_bar W_μ + W_bar]

# E[(out - μ)(out - μ)ᵀ], for independent marginals of `out` and `μ`, or for their joint, whose
# first half is `out` and second half `μ`.
function difference_moment(q_out, q_μ)
    m_out, V_out = mean_cov(q_out)
    m_μ, V_μ = mean_cov(q_μ)
    Δ = m_out - m_μ
    return V_out + V_μ + Δ * Δ'
end

function difference_moment(q_joint)
    m, V = mean_cov(q_joint)
    d = div(length(m), 2)
    Δ = @views m[1:d] - m[(d + 1):end]
    return @views V[1:d, 1:d] - V[1:d, (d + 1):end] - V[(d + 1):end, 1:d] + V[(d + 1):end, (d + 1):end] + Δ * Δ'
end

# The covariance a marginal `q_Σ` of a covariance contributes under naive VMP, E[Σ⁻¹]⁻¹; v6 used
# E[Σ] (ReactiveMP.jl#673, the multivariate form of #669). They agree for a point mass.
variational_covariance(q_Σ) = cholinv(mean(cholinv, q_Σ))

# The Gaussian message `m` through Gaussian noise of precision `Λ_f`: the mean of `m` and the
# precision (Λ⁻¹ + Λ_f⁻¹)⁻¹, computed as Λ - Λ (Λ + Λ_f)⁻¹ Λ with one Cholesky factorisation.
function series_precision(m, Λ_f)
    μ, Λ = mean_precision(m)
    return MvNormalMeanPrecision(μ, Λ - Λ * (fastcholesky(Λ + Λ_f) \ Λ))
end

# (d log 2π + rest) / 2, the average energy of a d-dimensional normal given the rest of it, in
# `rest`'s float type: `d * log2π` alone is a Float64 for an `Int` `d`, whatever the inputs.
gaussian_energy(d, rest) = (d * oftype(rest, log2π) + rest) / 2
