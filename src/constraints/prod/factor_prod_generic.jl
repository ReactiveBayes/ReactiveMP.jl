import Distributions
import Base: prod, show

"""
    FactorProduct
"""
struct FactorProduct{T}
    multipliers :: T
end

getmultipliers(product::FactorProduct)  = product.multipliers

Base.show(io::IO, product::FactorProduct) = print(io, "FactorProduct(", join(getmultipliers(product), ", ", ""), ")")

multipliers = (1, 2)
print(FactorProduct(multipliers))
