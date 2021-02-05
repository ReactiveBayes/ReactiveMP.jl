export GaussianMeanVariance, GaussianMeanPrecision, GaussianWeighteMeanPrecision
export MvGaussianMeanCovariance, MvGaussianMeanPrecision, MvGaussianWeightedMeanPrecision
export NormalDistributionsFamily, MvNormalDistributionsFamily
export UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily, NormalDistributionsFamily
export UnivariateGaussianDistributionsFamily, MultivariateGaussianDistributionsFamily, GaussianDistributionsFamily

const GaussianMeanVariance            = NormalMeanVariance
const GaussianMeanPrecision           = NormalMeanPrecision
const GaussianWeighteMeanPrecision    = NormalWeightedMeanPrecision
const MvGaussianMeanCovariance        = MvNormalMeanCovariance
const MvGaussianMeanPrecision         = MvNormalMeanPrecision
const MvGaussianWeightedMeanPrecision = MvNormalWeightedMeanPrecision

const UnivariateNormalDistributionsFamily{T}   = Union{NormalMeanPrecision{T}, NormalMeanVariance{T}, NormalWeightedMeanPrecision{T}}
const MultivariateNormalDistributionsFamily{T} = Union{MvNormalMeanPrecision{T}, MvNormalMeanCovariance{T}, MvNormalWeightedMeanPrecision{T}}
const NormalDistributionsFamily{T}             = Union{UnivariateNormalDistributionsFamily{T}, MultivariateNormalDistributionsFamily{T}}

const UnivariateGaussianDistributionsFamily   = UnivariateNormalDistributionsFamily
const MultivariateGaussianDistributionsFamily = MultivariateNormalDistributionsFamily
const GaussianDistributionsFamily             = NormalDistributionsFamily

import Base: prod, convert

# Conversion to mean - variance parametrisation

function Base.convert(::Type{ NormalMeanVariance{T} }, dist::UnivariateNormalDistributionsFamily) where { T <: Real } 
    return NormalMeanVariance(convert(T, mean(dist)), convert(T, var(dist)))
end

function Base.convert(::Type{ MvNormalMeanCovariance{T} }, dist::MultivariateNormalDistributionsFamily) where { T <: Real } 
    return MvNormalMeanCovariance(convert(AbstractArray{T}, mean(dist)), convert(AbstractArray{T}, cov(dist)))
end

function Base.convert(::Type{ NormalMeanVariance }, dist::UnivariateNormalDistributionsFamily{T}) where { T <: Real }
    return convert(NormalMeanVariance{T}, dist)
end

function Base.convert(::Type{ MvNormalMeanCovariance }, dist::MultivariateNormalDistributionsFamily{T}) where { T <: Real }
    return convert(MvNormalMeanCovariance{T}, dist)
end

# Conversion to mean - precision parametrisation

function Base.convert(::Type{ NormalMeanPrecision{T} }, dist::UnivariateNormalDistributionsFamily) where { T <: Real } 
    return NormalMeanPrecision(convert(T, mean(dist)), convert(T, precision(dist)))
end

function Base.convert(::Type{ MvNormalMeanPrecision{T} }, dist::MultivariateNormalDistributionsFamily) where { T <: Real } 
    return MvNormalMeanPrecision(convert(AbstractArray{T}, mean(dist)), convert(AbstractArray{T}, precision(dist)))
end

function Base.convert(::Type{ NormalMeanPrecision }, dist::UnivariateNormalDistributionsFamily{T}) where { T <: Real }
    return convert(NormalMeanPrecision{T}, dist)
end

function Base.convert(::Type{ MvNormalMeanPrecision }, dist::MultivariateNormalDistributionsFamily{T}) where { T <: Real }
    return convert(MvNormalMeanPrecision{T}, dist)
end

# Conversion to weighted mean - precision parametrisation

function Base.convert(::Type{ NormalWeightedMeanPrecision{T} }, dist::UnivariateNormalDistributionsFamily) where { T <: Real } 
    return NormalWeightedMeanPrecision(convert(T, weightedmean(dist)), convert(T, precision(dist)))
end

function Base.convert(::Type{ MvNormalWeightedMeanPrecision{T} }, dist::MultivariateNormalDistributionsFamily) where { T <: Real } 
    return MvNormalWeightedMeanPrecision(convert(AbstractArray{T}, weightedmean(dist)), convert(AbstractArray{T}, precision(dist)))
end

function Base.convert(::Type{ NormalWeightedMeanPrecision }, dist::UnivariateNormalDistributionsFamily{T}) where { T <: Real }
    return convert(NormalWeightedMeanPrecision{T}, dist)
end

function Base.convert(::Type{ MvNormalWeightedMeanPrecision }, dist::MultivariateNormalDistributionsFamily{T}) where { T <: Real }
    return convert(MvNormalWeightedMeanPrecision{T}, dist)
end

# Exstensions of prod methods

function Base.prod(::ProdPreserveParametrisation, left::NormalMeanVariance, right::UnivariateNormalDistributionsFamily)
    μ = (mean(left) * var(right) + mean(right) * var(left)) / (var(right) + var(left))
    v = (var(left) * var(right)) / (var(left) + var(right))
    return NormalMeanVariance(μ, v)
end

function Base.prod(::ProdPreserveParametrisation, left::NormalMeanPrecision, right::UnivariateNormalDistributionsFamily) 
    p = precision(left) + precision(right)
    μ = (mean(left) * precision(left) + mean(right) * precision(right)) / p
    return NormalMeanPrecision(μ, p)
end

# TODO add more prod implementations

# Basic prod fallbacks to weighted mean precision and converts first argument back

function Base.prod(::ProdBestSuitableParametrisation, left::L, right::R) where { L <: UnivariateNormalDistributionsFamily, R <: UnivariateNormalDistributionsFamily }
    wleft  = convert(NormalWeightedMeanPrecision, left)
    wright = convert(NormalWeightedMeanPrecision, right)
    return prod(ProdBestSuitableParametrisation(), wleft, wright) 
end

function Base.prod(::ProdBestSuitableParametrisation, left::L, right::R) where { L <: MultivariateNormalDistributionsFamily, R <: MultivariateNormalDistributionsFamily }
    wleft  = convert(MvNormalWeightedMeanPrecision, left)
    wright = convert(MvNormalWeightedMeanPrecision, right)
    return prod(ProdBestSuitableParametrisation(), wleft, wright) 
end

function Base.prod(::ProdPreserveParametrisation, left::L, right::R) where { L <: NormalDistributionsFamily, R <: NormalDistributionsFamily }
    return convert(L, prod(ProdBestSuitableParametrisation(), left, right))
end







