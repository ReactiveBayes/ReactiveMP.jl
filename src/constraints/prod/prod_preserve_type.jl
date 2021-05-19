export ProdPreserveType, ProdPreserveTypeLeft, ProdPreserveTypeRight

import Base: prod

"""
    ProdPreserveType{T}

`ProdPreserveType` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in some specific form.
By default it fallbacks to a `ProdAnalytical` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.

See also: [`prod`](@ref), [`ProdAnalytical`](@ref), [`ProdPreserveTypeLeft`](@ref), [`ProdPreserveTypeRight`](@ref)
"""
struct ProdPreserveType{T} end

ProdPreserveType(::Type{T}) where T = ProdPreserveType{T}()

prod(::ProdPreserveType{T}, left, right) where T = convert(T, prod(ProdAnalytical(), left, right))

"""
    ProdPreserveTypeLeft

`ProdPreserveTypeLeft` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in the functional form as `left` argument.
By default it fallbacks to a `ProdPreserveType` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdPreserveTypeRight`](@ref)
"""
struct ProdPreserveTypeLeft end

prod(::ProdPreserveTypeLeft, left::L, right) where L = prod(ProdPreserveType(L), left, right)

"""
    ProdPreserveTypeRight

`ProdPreserveTypeRight` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in the functional form as `right` argument.
By default it fallbacks to a `ProdPreserveType` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdPreserveTypeLeft`](@ref)
"""
struct ProdPreserveTypeRight end

prod(::ProdPreserveTypeRight, left, right::R) where R = prod(ProdPreserveType(R), left, right)