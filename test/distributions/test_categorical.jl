module CategoricalTest

using Test
using ReactiveMP
using Distributions
using Random
using StatsFuns

@testset "Categorical" begin

    # Categorical comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "vague" begin
        @test_throws MethodError vague(Categorical)

        d1 = vague(Categorical, 2)

        @test typeof(d1) <: Categorical
        @test probvec(d1) ≈ [0.5, 0.5]

        d2 = vague(Categorical, 4)

        @test typeof(d2) <: Categorical
        @test probvec(d2) ≈ [0.25, 0.25, 0.25, 0.25]
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Categorical([0.1, 0.4, 0.5]), Categorical([1 / 3, 1 / 3, 1 / 3])) == Categorical([0.1, 0.4, 0.5])
        @test prod(ProdAnalytical(), Categorical([0.1, 0.4, 0.5]), Categorical([0.8, 0.1, 0.1])) == Categorical([0.47058823529411764, 0.23529411764705882, 0.2941176470588235])
        @test prod(ProdAnalytical(), Categorical([0.2, 0.6, 0.2]), Categorical([0.8, 0.1, 0.1])) == Categorical([0.6666666666666666, 0.24999999999999994, 0.08333333333333333])
    end

    @testset "probvec" begin
        @test probvec(Categorical([0.1, 0.4, 0.5])) == [0.1, 0.4, 0.5]
        @test probvec(Categorical([1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
        @test probvec(Categorical([0.8, 0.1, 0.1])) == [0.8, 0.1, 0.1]
    end

    @testset "CategoricalNaturalParameters" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, CategoricalNaturalParameters([0 for _ in 1:(i - 1)])) ≈ Categorical([1 / i for _ in 1:i])
                @test convert(CategoricalNaturalParameters, [0 for _ in 1:i]) == CategoricalNaturalParameters([0 for _ in 1:i])
                @test convert(CategoricalNaturalParameters{Float64}, [0 for _ in 1:i]) == CategoricalNaturalParameters([0 for _ in 1:i])
                @test as_naturalparams(CategoricalNaturalParameters, [0 for _ in 1:i]) == CategoricalNaturalParameters([0 for _ in 1:i])
                @test naturalparams(Categorical([1 / i for _ in 1:i])) == CategoricalNaturalParameters([0 for _ in 1:(i - 1)])
            end
        end

        @testset "logpdf" begin
            for i in 1:10
                distribution = Categorical(softmax([rand() for _ in 1:i]))
                cat_np = naturalparams(distribution)
                for j in 1:i
                    @test logpdf(distribution, j) ≈ logpdf(cat_np, j)
                end
            end
        end

        @testset "lognormalizer" begin
            for i in 1:10
                distribution = Categorical(softmax([rand() for _ in 1:i]))
                cat_np = naturalparams(distribution)
                @test lognormalizer(cat_np) ≈ log(sum(exp.(cat_np.η)) + 1)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(CategoricalNaturalParameters([rand() for _ in 1:i])) === true
            end
        end
    end
end

end
