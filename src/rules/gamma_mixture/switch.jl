using .GammaMixtureHelpers

@rule GammaMixture{N}(:switch, Marginalisation) (q_out::Any, q_a::NTuple{N, GammaDistributionsFamily }, q_b::NTuple{N, GammaDistributionsFamily }) where { N } = begin
    # p. 12
    ρ =  map(1:N) do k
        x = Λ(q_b[k]) * mean(q_a[k])
        y = Λ(q_out) * (mean(q_a[k]) - 1)
        z = Λ(q_out) * mean(q_a[k])
        d = mean(q_b[k]) * mean(q_out)
        return exp(x + y - z - d)
    end
    return Categorical(ρ ./ sum(ρ))
end
