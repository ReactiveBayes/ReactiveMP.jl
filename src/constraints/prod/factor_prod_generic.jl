export FactorProduct
import Distributions: entropy
import Base: getindex, show, length

"""
    FactorProduct

If a rule needs to send a density of the form d1d2...di...dn, it may use `FactorProduct` structure.
You can create FactorProduct in the following way `FactorProduct((d1, d2, ...))`: the constructor expects to obtain a tuple.
It is supposed that `entropy` is implemented for the tuple elements (d1, d2, ...): entropy of `FactorProduct` is the sum of input entropies.
You can access each individual (di) from a factor product (`factor_product = FactorProduct((d1, d2,..., di, ...))`) in the following way `factor_product[i]`.
"""
struct FactorProduct{T}
    multipliers::T
end

getmultipliers(product::FactorProduct) = product.multipliers

Base.show(io::IO, product::FactorProduct) = print(io, "FactorProduct(", join(getmultipliers(product), ", ", ", "), ")")

Base.getindex(product::FactorProduct, i::Int) = product.multipliers[i]

Base.length(product::FactorProduct) = length(product.multipliers)

Distributions.entropy(product::FactorProduct) = mapreduce(entropy, +, getmultipliers(product))
