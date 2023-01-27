module ForwardDiffGradTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions
using ForwardDiff

import ReactiveMP: convert_eltype

@testset "ForwardDiffGrad" begin
    grad = ForwardDiffGrad()

    check_basic_statistics =
        (left, right, dims) -> begin
            for value in (fill(1.0, dims), fill(-1.0, dims), fill(0.0, dims), mean(left), mean(right), rand(dims))
                @test all(isapprox.(ReactiveMP.compute_gradient(grad, (x) -> logpdf(left, x), value), ReactiveMP.compute_gradient(grad, (x) -> logpdf(left, x), value), atol = 1e-14))
                @test all(isapprox.(ReactiveMP.compute_hessian(grad, (x) -> logpdf(left, x), value), ReactiveMP.compute_hessian(grad, (x) -> logpdf(left, x), value), atol = 1e-14))
            end
        end

    types  = ReactiveMP.union_types(MultivariateNormalDistributionsFamily{Float64})
    etypes = ReactiveMP.union_types(MultivariateNormalDistributionsFamily)

    dims = (2, 3, 5)
    rng  = MersenneTwister(1234)

    for dim in dims
        for type in types
            left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(rand(rng, Float64, dim))))
            check_basic_statistics(left, convert(MvNormal, left), dim)
            for type in [types..., etypes...]
                right = convert(type, left)
                check_basic_statistics(left, right, dim)

                p1 = prod(ProdPreserveTypeLeft(), left, right)
                @test typeof(p1) <: typeof(left)

                p2 = prod(ProdPreserveTypeRight(), left, right)
                @test typeof(p2) <: typeof(right)

                p3 = prod(ProdAnalytical(), left, right)

                check_basic_statistics(p1, p2, dim)
                check_basic_statistics(p2, p3, dim)
                check_basic_statistics(p1, p3, dim)
            end
        end
    end
end

end
