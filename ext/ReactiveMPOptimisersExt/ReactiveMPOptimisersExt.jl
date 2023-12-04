module ReactiveMPOptimisersExt

using ReactiveMP, Optimisers

# `Optimisers.jl` export an `AbstractRule` and a `Leaf`

function ReactiveMP.cvi_setup(opt::Optimisers.AbstractRule, λ)
    # We rely on the `deepcopy` here, because some optimizers from the `Optimizers.jl` may change their state
    copt = deepcopy(opt)
    init = Optimisers.init(copt, λ)
    return (copt, init)
end

function ReactiveMP.cvi_update!(opt_and_state::Tuple{Optimisers.AbstractRule, Any}, new_λ, λ, ∇)
    # Retrieve the optimiser and its current state
    optimiser, current_state = opt_and_state
    # Apply the optimiser to the current state and adjust the gradient
    adjusted_state, adjusted_∇ = Optimisers.apply!(optimiser, current_state, λ, ∇)
    # Update the vector of parameters λ
    @inbounds for (i, λᵢ, Δᵢ) in zip(eachindex(new_λ), λ, adjusted_∇)
        new_λ[i] = λᵢ - Δᵢ
    end
    return (optimiser, adjusted_state), new_λ
end

end
