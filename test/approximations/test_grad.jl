module ForwardDiffGradTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions
using ForwardDiff

@testset "ForwardDiffGrad" begin
    grad = ForwardDiffGrad()

    for i in 1:100
        @test ReactiveMP.compute_gradient(grad, (x) -> sum(x)^2, [i]) ≈ [2 * i]
        @test ReactiveMP.compute_hessian(grad, (x) -> sum(x)^2, [i]) ≈ [2;;]
    end
end

end
