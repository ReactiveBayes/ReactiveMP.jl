sampletype(obj::Any) = error("sampletype is not implemented for $(typeof(obj))")
samplefloattype(obj) = deep_eltype(sampletype(obj))
variate_form2type(::Type{Univariate}, ::Type{T}) where {T} = T
variate_form2type(::Type{Multivariate}, ::Type{T}) where {T} = Vector{T}
variate_form2type(::Type{Matrixvariate}, ::Type{T}) where {T} = Matrix{T}
sampletype(dist::Distribution) = variate_form2type(variate_form(dist), eltype(dist))
