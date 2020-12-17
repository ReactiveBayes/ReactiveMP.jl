export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::Type{T}, ::VariableBoundEntropy, variable::RandomVariable, scheduler) where { T <: InfCountingReal }
    mapping = let d = degree(variable)
        (m) -> convert(T, (d - 1) * score(DifferentialEntropy(), m))
    end
    return getmarginal(variable) |> schedule_on(scheduler) |> map(T, mapping)
end