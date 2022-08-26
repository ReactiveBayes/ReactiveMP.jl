export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::Type{T}, ::VariableBoundEntropy, variable::RandomVariable, skip_strategy, scheduler) where {T <: InfCountingReal}
    mapping = let d = degree(variable)
        (marginal) -> convert(T, (d - 1) * score(DifferentialEntropy(), marginal))
    end
    return getmarginal(variable, skip_strategy) |> schedule_on(scheduler) |> map(T, mapping)
end
