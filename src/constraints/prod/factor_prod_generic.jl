export FactorProduct

import Distributions: entropy
import Base: getindex, show, length

"""
    FactorProduct
"""
struct FactorProduct{T}
    multipliers::T
end

getmultipliers(product::FactorProduct) = product.multipliers

Base.show(io::IO, product::FactorProduct) = print(io, "FactorProduct(", join(getmultipliers(product), ", ", ", "), ")")

Base.getindex(product::FactorProduct, i::Int) = product.multipliers[i]

Base.length(product::FactorProduct) = length(product.multipliers)

Distributions.entropy(product::FactorProduct) = mapreduce(entropy, +, getmultipliers(product))

