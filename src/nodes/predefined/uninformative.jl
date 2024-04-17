export Uninformative

struct Uninformative end

@node Uninformative Stochastic [out]

@average_energy Uninformative (q_out::Any,) = entropy(q_out)

function BayesBase.default_prod_rule(::Type{<:Uninformative}, ::Type{T}) where {T}
    return PreserveTypeProd(T)
end

function BayesBase.default_prod_rule(::Type{T}, ::Type{<:Uninformative}) where {T}
    return PreserveTypeProd(T)
end

function BayesBase.default_prod_rule(::Type{<:Uninformative}, ::Type{<:Uninformative})
    return PreserveTypeProd(Uninformative)
end

function Base.prod(::PreserveTypeProd{T}, left::Uninformative, right::T) where {T}
    return right
end

function Base.prod(::PreserveTypeProd{T}, left::T, right::Uninformative) where {T}
    return left
end

function Base.prod(::PreserveTypeProd{Uninformative}, left::Uninformative, right::Uninformative)
    return Uninformative()
end
