export MvNormalMeanScalePrecision

struct MvNormalMeanScalePrecision{T <: Real, M <: AbstractVector{T}} <: AbstractMvNormal
    μ::M
    γ::T
end
