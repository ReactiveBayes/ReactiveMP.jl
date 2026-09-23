export score, DifferentialEntropy

function score end

##

struct DifferentialEntropy end

struct KLDivergence end

## Differential entropy function helpers

score(::DifferentialEntropy, marginal::Marginal) = entropy(marginal)

function score(::DifferentialEntropy, marginal::Marginal{<:NamedTuple})
    compute_score =
    let is_marginal_clamped = is_clamped(marginal),
            is_marginal_initial = is_initial(marginal)

        (data) -> score(
            DifferentialEntropy(),
            Marginal(data, is_marginal_clamped, is_marginal_initial),
        )
    end

    return mapreduce(compute_score, +, values(getdata(marginal)))
end

## Kl KlDivergence

score(::KLDivergence, marginal::Marginal, p::Distribution) =
    Distributions.kldivergence(getdata(marginal), p)
