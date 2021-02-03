
function prod(::ProdPreserveParametrisation, left::Bernoulli, right::Categorical)
    @assert length(probvec(right)) === 2 "Improper Bernoulli x Categorical product"

    left_p  = succprob(left)
    right_p = first(probvec(right))

    pprod = left_p * right_p 
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end