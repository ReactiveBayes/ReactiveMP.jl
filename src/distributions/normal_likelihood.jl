using Distributions
using Optim


struct NormalLikelihood{T <: Function, A}
    logp :: T

    approximation :: A
end

Distributions.logpdf(distribution::NormalLikelihood, x) = distribution.logp(x)

function approximate_prod_expectations(approximation::LaplaceApproximation, left::NormalDistributionsFamily, right::NormalLikelihood)

    m, v = approximate_meancov(approximation, right.logp, left)

    return m, v
end

function prod(::ProdPreserveParametrisation, left::NormalLikelihood, right::NormalDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::NormalDistributionsFamily, right::NormalLikelihood)
    
    ms, vs = mean(left), cov(left)

    logf = (x) -> right.logp(x) - 1/2/vs*(x-ms)^2

    minms = Optim.minimizer(optimize(x->-logf(first(x)), [ms], LBFGS()))[1]
    minvs = ForwardDiff.hessian(x->-logf(first(x)), [minms])[1,1]

    if minvs < 0
        println(minms, minvs)
        minvs = 1e20
    end

    return NormalMeanPrecision(minms, 1/minvs)
end
