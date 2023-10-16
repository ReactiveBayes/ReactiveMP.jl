module ReactiveMPOptimisersExt

using ReactiveMP, Optimisers

# `Optimisers.jl` export an `AbstractRule` and a `Leaf`

function ReactiveMP.cvi_setup(opt::Optimisers.AbstractRule, λ)
    # We rely on the `deepcopy` here, because some optimizers from the `Optimizers.jl` may change their state
    copt = deepcopy(opt)
    init = Optimisers.init(copt, vec(λ))
    return (copt, init)
end

function ReactiveMP.cvi_update!(opt_and_state::Tuple{Optimisers.AbstractRule, Any}, λ, ∇)
    opt, state = opt_and_state
    new_state, new_∇ = Optimisers.apply!(opt, state, vec(λ), vec(∇))
    return (opt, new_state), new_∇
end

end
