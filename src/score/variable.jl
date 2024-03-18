export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::Type{T}, ::VariableBoundEntropy, variable::RandomVariable, skip_strategy, scheduler) where {T <: CountingReal}
    mapping = let d = degree(variable)
        (marginal) -> begin
            # The entropy of point masses is not finite
            entropy = convert(T, score(DifferentialEntropy(), marginal))
            # In this case we treat them as clamped variables, such that we should multiply 
            # their influence on `d` instead of `d - 1`
            scaling = isfinite(entropy) ? (d - 1) : d
            return scaling * entropy
        end
    end
    return getmarginal(variable, skip_strategy) |> schedule_on(scheduler) |> map(T, mapping)
end
