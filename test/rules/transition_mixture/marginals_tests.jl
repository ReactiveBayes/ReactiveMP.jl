@testitem "marginalrules:TransitionMixture" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules
    @testset "out_in_switch: (m_out::Categorical, m_in::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, Union{PointMass, MatrixDirichlet}})" begin
        @test ReactiveMP.marginalrule(
            TransitionMixture{2},
            Val(:out_in_switch),
            Val((:out, :in, :switch)),
            (
                ReactiveMP.Message(Categorical(0.2, 0.8), false, true, nothing),
                ReactiveMP.Message(Categorical(0.1, 0.9), false, true, nothing),
                ReactiveMP.Message(Categorical(0.3, 0.7), false, true, nothing)
            ),
            Val((:matrices_1, :matrices_2)),
            (ReactiveMP.Marginal(PointMass([0.1 0.2; 0.9 0.8]), false, true, nothing), ReactiveMP.Marginal(PointMass([0.3 0.4; 0.7 0.6]), false, true, nothing)),
            Nothing(),
            Nothing()
        ) ≈ Contingency(
            permutedims(
                stack([
                    [0.0009966777408637875 0.017940199335548173; 0.035880398671096346 0.28704318936877077],
                    [0.0069767441860465115 0.08372093023255814; 0.06511627906976744 0.5023255813953489]
                ]),
                (3, 1, 2)
            )
        )

        @test ReactiveMP.marginalrule(
            TransitionMixture{2},
            Val(:out_in_switch),
            Val((:out, :in, :switch)),
            (
                ReactiveMP.Message(Categorical(0.2, 0.4, 0.4), false, true, nothing),
                ReactiveMP.Message(Categorical(0.1, 0.5, 0.4), false, true, nothing),
                ReactiveMP.Message(Categorical(0.3, 0.7), false, true, nothing)
            ),
            Val((:matrices_1, :matrices_2)),
            (
                ReactiveMP.Marginal(
                    MatrixDirichlet(
                        [
                            3.759109596106528 7.651848771355311 9.463024296540873
                            0.05227997457759148 7.63832734893159 4.123039206788142
                            1.7218244939296623 0.8732644376397225 5.797052273450543
                        ]
                    ),
                    false,
                    true,
                    nothing
                ),
                ReactiveMP.Marginal(
                    PointMass(
                        [
                            9.712482722213574 9.112005385625451 9.022921164635415
                            2.158924800523061 9.014480452856436 7.641284931051801
                            2.3506858907854378 5.462580545036949 5.43795594079331
                        ]
                    ),
                    false,
                    true,
                    nothing
                )
            ),
            Nothing(),
            Nothing()
        ) ≈ Contingency{Float64, Array{Float64, 3}}(
            [
                0.0007817591287544585 1.4372761803507031e-12 0.0005989009423266854; 0.027299775547839533 0.012136580144260207 0.013214581490095035;;;
                0.0027518963746678685 0.0054934040277940565 0.00034292322803211974; 0.12805979116407934 0.2533783586358652 0.15354181526669014;;;
                0.0022880577360659344 0.0018546054978035274 0.002707028738712131; 0.10144624381859718 0.17182454330668553 0.12227973495029375
            ]
        )
    end
end
