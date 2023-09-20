module ReactiveMPOptimisersExt

using ReactiveMP, Optimisers

# `Optimisers.jl` export an `AbstractRule` and a `Leaf`

function ReactiveMP.cvi_setup(opt::Optimisers.AbstractRule, λ)
    # We rely on the `deepcopy` here, because some optimizers from the `Optimizers.jl` may change their state
    return Optimisers.setup(deepcopy(opt), vec(λ))
end

function ReactiveMP.cvi_update!(opt::Optimisers.Leaf, λ, ∇)
    # I'm not sure we can ignore the updated tree?
    _, result = Optimisers.update!(opt, vec(λ), vec(∇))
    return result
end

end
