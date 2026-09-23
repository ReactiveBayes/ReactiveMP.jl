# MessagePassingRulesApproximations against v6's `src/approximations/`, on identical inputs.
# The multi-input unscented path matters most: it builds its joint from concatenated means
# and a block-diagonal covariance, where v6 went through ExponentialFamily's JointNormal.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_approximations.jl

using Test, LinearAlgebra
import ReactiveMP
import MessagePassingRulesApproximations as Approximations

agree(a, b) = a === b || (a isa Tuple && b isa Tuple && length(a) == length(b) && all(map(agree, a, b))) ||
    (a isa Union{Number, AbstractArray} && b isa Union{Number, AbstractArray} && size(a) == size(b) && isapprox(a, b; atol = 1.0e-12))

square(x) = x^2 + 1.0
stack3(x) = [x[1]^2, x[1] * x[2], sin(x[2])]
mixed(a, b, c) = [a * b[1], a + c, b[2] * c]

const UNSCENTED_CASES = [
    ("univariate", square, (0.5,), (2.0,)),
    ("univariate, vector output", x -> [x^2, x], (1.0,), (0.5,)),
    ("multivariate", stack3, ([0.5, -1.0],), ([1.0 0.3; 0.3 2.0],)),
    ("two scalars", (a, b) -> a * b + b, (0.5, -1.0), (2.0, 0.25)),
    ("scalar, vector and scalar", mixed, (0.5, [1.0, -2.0], 3.0), (2.0, [1.0 0.2; 0.2 0.5], 0.25)),
]

@testset "MessagePassingRulesApproximations against v6" begin
    for method in (Approximations.Unscented(), Approximations.Unscented(; alpha = 0.5, beta = 1.5, kappa = 1.0))
        v6_method = ReactiveMP.Unscented(; alpha = method.α, beta = method.β, kappa = method.κ)
        for (label, f, means, covs) in UNSCENTED_CASES
            @testset "unscented_statistics: $label, α = $(method.α)" begin
                @test agree(Approximations.unscented_statistics(method, f, means, covs), ReactiveMP.unscented_statistics(v6_method, f, means, covs))
                @test agree(Approximations.approximate(method, f, means, covs), ReactiveMP.approximate(v6_method, f, means, covs))
            end
        end
    end
    @testset "smoothRTS" begin
        @test agree(Approximations.smoothRTS(4.0, 1.5, 0.3, 2.0, 3.0, 5.0, 1.0), ReactiveMP.smoothRTS(4.0, 1.5, 0.3, 2.0, 3.0, 5.0, 1.0))
        args = ([1.0, 2.0], [2.0 0.3; 0.3 1.0], [0.5 0.1; 0.2 0.4], [0.5, 1.5], [1.0 0.0; 0.0 2.0], [1.5, 2.5], [0.5 0.1; 0.1 0.5])
        @test agree(Approximations.smoothRTS(args...), ReactiveMP.smoothRTS(args...))
    end
end
