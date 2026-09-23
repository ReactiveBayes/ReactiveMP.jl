"""
    isonehot(vec::AbstractVector)

Whether `vec` has exactly one entry approximately equal to one and all others approximately
zero, to within `sqrt(eps)` of its element type.
"""
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

"""
    promoted_cluster(cluster::FactorizedCluster, inputs...)

`cluster` with every block in the float type of all `inputs` together. A rule's output must
carry the promoted float type of every input, and that includes a block that passes an input
through unchanged, such as v6's `v = m_v`.
"""
promoted_cluster(cluster::FactorizedCluster, inputs...) =
    BayesBase.convert_paramfloattype(BayesBase.promote_paramfloattype(inputs...), cluster)
