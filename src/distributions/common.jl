
function prod(::ProdPreserveParametrisation, left::Bernoulli, right::Categorical)
    @assert length(probvec(right)) === 2 "Improper Bernoulli x Categorical product"
    return prod(ProdPreserveParametrisation(), left, Bernoulli(first(probvec(right))))
end