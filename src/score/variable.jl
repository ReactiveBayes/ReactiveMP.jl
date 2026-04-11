export VariableBoundEntropy

"""
    VariableBoundEntropy

Dispatch tag for computing the scaled entropy contribution of a random variable
node to the Bethe free energy: `(d - 1) H[q]`, where `d` is the degree (number of
connected factor nodes) and `H[q]` is the marginal entropy. Used as the first
argument to [`score`](@ref).
"""
struct VariableBoundEntropy end

function score(
    ::Type{T}, ::VariableBoundEntropy, variable::RandomVariable, scheduler
) where {T <: CountingReal}
    mapping = let d = degree(variable)
        (marginal) -> begin
            # The entropy of point masses is not finite
            # In this case we treat them as clamped variables, such that we should multiply
            # their influence on `d` instead of `d - 1`
            scaling = !ispointmass(marginal) ? (d - 1) : d
            entropy = convert(T, score(DifferentialEntropy(), marginal))
            return scaling * entropy
        end
    end
    return get_stream_of_marginals(variable) |>
           skip_initial() |>
           schedule_on(scheduler) |>
           map(T, mapping)
end
