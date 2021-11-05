export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::Type{T}, objective::BetheFreeEnergy, ::VariableBoundEntropy, variable::RandomVariable, scheduler) where { T <: InfCountingReal }
    mapping = let d = degree(variable)
        (marginal) -> convert(T, (d - 1) * score(DifferentialEntropy(), marginal))
    end
    return getmarginal(variable, marginal_skip_strategy(objective)) |> schedule_on(scheduler) |> map(T, mapping)
end
