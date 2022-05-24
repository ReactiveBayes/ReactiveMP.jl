module ReactiveMPProdFinalTest

using Test
using ReactiveMP
using Random
using LinearAlgebra

import ReactiveMP: getdist

@testset "ProdFinal" begin
    uni_distributions = [
        Gamma(1.0, 2.0),
        NormalMeanVariance(-10.0, 3.0)
    ]

    mv_distributions = [
        MvNormalMeanCovariance(ones(3)),
        MvNormalMeanPrecision(3ones(3))
    ]

    mxv_distributions = [
        Wishart(3, Matrix(Diagonal(ones(3)))),
        Wishart(4, Matrix(Diagonal(ones(4))))
    ]

    sets = (uni_distributions, mv_distributions, mxv_distributions)

    for dset in sets
        for i in 1:length(dset), j in 1:length(dset)
            left = dset[i]
            right = dset[j]

            @test (@inferred getdist(prod(ProdAnalytical(), ProdFinal(left), right))) === left
            @test (@inferred getdist(prod(ProdAnalytical(), left, ProdFinal(right)))) === right
            @test_throws ErrorException prod(ProdAnalytical(), ProdFinal(left), ProdFinal(right))
        end
    end

    # Check errors for different variate_forms
    for i in 1:length(sets), j in 1:length(sets)
        if i !== j
            dset1 = sets[i]
            dset2 = sets[j]

            for dist1 in dset1, dist2 in dset2
                @test_throws ErrorException prod(ProdAnalytical(), ProdFinal(dist1), dist2)
                @test_throws ErrorException prod(ProdAnalytical(), dist1, ProdFinal(dist2))
            end
        end
    end
end

end
