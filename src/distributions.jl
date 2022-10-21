export vague
export mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov
export mean_cov, mean_var, mean_std, mean_invcov, mean_precision, weightedmean_cov, weightedmean_var, weightedmean_std, weightedmean_invcov, weightedmean_precision
export weightedmean, probvec
export variate_form, value_support, promote_variate_type, convert_eltype

import Distributions: mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov
import Distributions: VariateForm, ValueSupport, Distribution

"""
    vague(distribution_type, [ dims... ])

`vague` function returns uninformative probability distribution of a given type and can be used to set an uninformative priors in a model.
"""
function vague end

mean_cov(something)       = (mean(something), cov(something))
mean_var(something)       = (mean(something), var(something))
mean_std(something)       = (mean(something), std(something))
mean_invcov(something)    = (mean(something), invcov(something))
mean_precision(something) = mean_invcov(something)

weightedmean_cov(something)       = (weightedmean(something), cov(something))
weightedmean_var(something)       = (weightedmean(something), var(something))
weightedmean_std(something)       = (weightedmean(something), std(something))
weightedmean_invcov(something)    = (weightedmean(something), invcov(something))
weightedmean_precision(something) = weightedmean_invcov(something)

probvec(something)      = error("Probability vector function probvec() is not defined for $(something)")
weightedmean(something) = error("Weighted mean is not defined for $(something)")

"""
    variate_form(distribution_or_type)

Returns the `VariateForm` sub-type (defined in `Distributions.jl`):

- `Univariate`, a scalar number
- `Multivariate`, a numeric vector
- `Matrixvariate`, a numeric matrix

Note: supports real-valued containers, for which it defines:

- `variate_form(real) = Univariate`
- `variate_form(vector_of_reals) = Multivariate`
- `variate_form(matrix_of_reals) = Matrixvariate`

See also: [`ReactiveMP.value_support`](@ref)
"""
variate_form(::Distribution{F, S}) where {F <: VariateForm, S <: ValueSupport} = F
variate_form(::Type{<:Distribution{F, S}}) where {F <: VariateForm, S <: ValueSupport} = F

variate_form(::Type{T}) where {T <: Real} = Univariate
variate_form(::T) where {T <: Real}       = Univariate

variate_form(::Type{V}) where {T <: Real, V <: AbstractVector{T}} = Multivariate
variate_form(::V) where {T <: Real, V <: AbstractVector{T}}       = Multivariate

variate_form(::Type{M}) where {T <: Real, M <: AbstractMatrix{T}} = Matrixvariate
variate_form(::M) where {T <: Real, M <: AbstractMatrix{T}}       = Matrixvariate

# Note that the recent version of `Distributions.jl` has the exact same methods (`variate_form`) with the exact same names, however, old versions do not.
# We keep that for backward-compatibility with old `Distributions.jl` versions,
# but probably we should revise this at some point and remove our implementations (except for the `real` constrained versions)

"""
    value_support(distribution_or_type)

Returns the `ValueSupport` sub-type (defined in `Distributions.jl`):

- `Discrete`, samples take discrete values
- `Continuous`, samples take continuous real values

See also: [`ReactiveMP.variate_form`](@ref)
"""
value_support(::Distribution{F, S}) where {F <: VariateForm, S <: ValueSupport} = S
value_support(::Type{<:Distribution{F, S}}) where {F <: VariateForm, S <: ValueSupport} = S

# Note that the recent version of `Distributions.jl` has the exact same methods (`value_support`) with the exact same names, however, old versions do not
# We keep that for backward-compatibility with old `Distributions.jl` versions, but probably we should revise this at some point and remove our implementations

"""
    promote_variate_type(::Type{ <: VariateForm }, distribution_type)

Promotes (if possible) a `distribution_type` to be of the specified variate form.
"""
function promote_variate_type end

promote_variate_type(::D, T) where {D <: Distribution}       = promote_variate_type(variate_form(D), T)
promote_variate_type(::Type{D}, T) where {D <: Distribution} = promote_variate_type(variate_form(D), T)

"""
    convert_eltype(::Type{D}, ::Type{E}, distribution)

Converts (if possible) a `distribution` to be of type `D{E}`.
"""
function convert_eltype end

convert_eltype(::Type{D}, ::Type{E}, distribution::Distribution) where {D <: Distribution, E} = convert(D{E}, distribution)

"""
    logpdf_sample_friendly(distribution) 
    
`logpdf_sample_friendly` function takes as an input a `distribution` and returns corresponding optimized two versions 
for taking `logpdf()` and sampling with `rand!` respectively. By default returns the same distribution, but some distributions 
may override default behaviour for better efficiency.

# Example

```jldoctest
julia> d = vague(MvNormalMeanPrecision, 2)
MvNormalMeanPrecision(
μ: [0.0, 0.0]
Λ: [1.0e-12 0.0; 0.0 1.0e-12]
)


julia> ReactiveMP.logpdf_sample_friendly(d)
(FullNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0e12 -0.0; -0.0 1.0e12]
)
, FullNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0e12 -0.0; -0.0 1.0e12]
)
)
```
"""
logpdf_sample_friendly(something) = (something, something)
