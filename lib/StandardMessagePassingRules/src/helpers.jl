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
