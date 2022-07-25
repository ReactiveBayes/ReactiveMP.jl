export FactorProduct
import Distributions
import Base: getindex, show, length

"""
    FactorProduct
"""
struct FactorProduct{T}
    multipliers::T
end

getmultipliers(product::FactorProduct) = product.multipliers

Base.show(io::IO, product::FactorProduct) = print(io, "FactorProduct(", join(getmultipliers(product), ", ", ", "), ")")

getindex(product::FactorProduct, i::Int) = product.multipliers[i]

length(product::FactorProduct) = length(product.multipliers)
