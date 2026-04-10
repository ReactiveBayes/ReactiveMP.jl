export VariableBoundEntropy

struct VariableBoundEntropy end

function score(
    ::Type{T},
    ::VariableBoundEntropy,
    variable::RandomVariable,
    scheduler,
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
