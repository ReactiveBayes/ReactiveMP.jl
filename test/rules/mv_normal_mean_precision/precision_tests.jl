
@testitem "rules:MvNormalMeanPrecision:precision" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    using FastCholesky

    import ExponentialFamily: WishartFast
    import ReactiveMP: @test_rules

    @testset "Variational: (q_out::PointMass, q_μ::MultivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (
                input = (q_out = PointMass([1.0, 2.0]), q_μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),
                output = WishartFast(4.0, cholinv([75/73 -46/73; -46/73 36/73]))
            ),
            (
                input = (q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])),
                output = WishartFast(4.0, cholinv([75/73 -46/73; -46/73 36/73]))
            ),
            (
                input = (q_out = MvNormalMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = MvNormalMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])),
                output = WishartFast(4.0, cholinv([39/74 -11/37; -11/37 10/37]))
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]),
                    q_μ = MvNormalMeanPrecision([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])
                ),
                output = WishartFast(5.0, cholinv([15/11 -3/22 5/11; -3/22 19/22 5/11; 5/11 5/11 16/33]))
            )
        ]
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([1.0, 2.0]), q_μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0])), output = WishartFast(4.0, cholinv([13/27 -8/27; -8/27 7/27]))),
            (input = (q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])), output = WishartFast(4.0, cholinv([13/27 -8/27; -8/27 7/27]))),
            (
                input = (q_out = MvNormalMeanCovariance([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = MvNormalMeanCovariance([3.0; 5.0], [3.0 2.0; 2.0 4.0])),
                output = WishartFast(4.0, cholinv([17/70 -1/7; -1/7 1/7]))
            ),
            (
                input = (
                    q_out = MvNormalMeanCovariance([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]),
                    q_μ = MvNormalMeanCovariance([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])
                ),
                output = WishartFast(5.0, cholinv([51/338 -6/169 5/169; -6/169 115/676 45/676; 5/169 45/676 47/676]))
            )
        ]
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (
                input = (q_out = PointMass([1.0, 2.0]), q_μ = MvNormalWeightedMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0])),
                output = WishartFast(4.0, cholinv([73/67 -26/67; -26/67 68/67]))
            ),
            (
                input = (q_out = MvNormalWeightedMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = PointMass([3.0; 5.0])),
                output = WishartFast(4.0, cholinv([165/163 -106/163; -106/163 76/163]))
            ),
            (
                input = (q_out = MvNormalWeightedMeanPrecision([1.0; 2.0], [3.0 2.0; 2.0 4.0]), q_μ = MvNormalWeightedMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])),
                output = WishartFast(4.0, cholinv([73/70 11/35; 11/35 34/35]))
            ),
            (
                input = (
                    q_out = MvNormalWeightedMeanPrecision([1.0; 2.0; 3.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0]),
                    q_μ = MvNormalWeightedMeanPrecision([3.0; 5.0; -2.0], [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])
                ),
                output = WishartFast(5.0, cholinv([459/338 -36/169 60/169; -36/169 115/169 90/169; 60/169 90/169 188/169]))
            )
        ]
    end

    @testset "Variational: (q_out_μ::MultivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance(ones(4), diageye(4)),), output = WishartFast(4.0, cholinv([0.5 0.0; 0.0 0.5]))),
            (input = (q_out_μ = MvNormalMeanPrecision(ones(4), diageye(4)),), output = WishartFast(4.0, cholinv([0.5 0.0; 0.0 0.5]))),
            (input = (q_out_μ = MvNormalWeightedMeanPrecision(ones(4), diageye(4)),), output = WishartFast(4.0, cholinv([0.5 0.0; 0.0 0.5])))
        ]

        a = [1.0, 2.0, -1.0, -2.0]
        A = [3.5 -0.5 -0.25 0.0; -0.5 3.0 -0.25 0.0; -0.25 -0.25 6.0 0.25; 0.0 0.0 0.25 7.0]

        @test_rules [check_type_promotion = true] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance(a, A),), output = WishartFast(4.0, cholinv([13/150 -2/75; -2/75 7/150]))),
            (input = (q_out_μ = MvNormalMeanPrecision(a, A),), output = WishartFast(4.0, cholinv([3751/1966 -3653/3932; -3653/3932 30259/58980]))),
            (
                input = (q_out_μ = MvNormalWeightedMeanPrecision(a, A),),
                output = WishartFast(4.0, cholinv([139109613/68523766 -96827813/137047532; -96827813/137047532 62742851/68523766]))
            )
        ]
    end
end
