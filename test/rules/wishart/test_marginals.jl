module RulesWishartMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using FastCholesky

import ExponentialFamily: WishartFast, @test_marginalrules

@testset "marginalrules:Wishart" begin
    @testset ":out_ν_S (m_out::Wishart, m_ν::PointMass, m_S::PointMass)" begin
        @test_marginalrules [check_type_promotion = true] Wishart(:out_ν_S) [
            (
                input = (m_out = WishartFast(3.0, cholinv([3.0 -1.0; -1.0 4.0])), m_ν = PointMass(2.0), m_S = PointMass([1.0 0.0; 0.0 1.0])),
                output = (out = Wishart(2.0, [14/19 -1/19; -1/19 15/19]), ν = PointMass(2.0), S = PointMass([1.0 0.0; 0.0 1.0]))
            ),
            (
                input = (m_out = WishartFast(7.0, cholinv([9.0 -2.0; -2.0 1.0])), m_ν = PointMass(4.0), m_S = PointMass([4.0 -2.0; -2.0 4.0])),
                output = (out = Wishart(8.0, [128/49 -34/49; -34/49 32/49]), ν = PointMass(4.0), S = PointMass([4.0 -2.0; -2.0 4.0]))
            ),
            (
                input = (
                    m_out = WishartFast(4.0, cholinv([9.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 11.0])),
                    m_ν = PointMass(3.0),
                    m_S = PointMass([11.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 9.0])
                ),
                output = (
                    out = Wishart(3.0, [2092/423 -1/1 211/423; -1/1 5/2 -1/1; 211/423 -1/1 2092/423]),
                    ν = PointMass(3.0),
                    S = PointMass([11.0 -2.0 1.0; -2.0 5.0 -2.0; 1.0 -2.0 9.0])
                )
            )
        ]
    end
end

end
