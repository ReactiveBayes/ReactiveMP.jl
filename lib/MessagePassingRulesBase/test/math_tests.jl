# The math helpers the rule packages share.

@testitem "math:linear algebra" tags = [:base] begin
    import MessagePassingRulesBase as B
    using LinearAlgebra

    A = [2.0 1.0; 0.5 3.0]
    x, y = [1.0, -2.0], [0.5, 4.0]
    for T1 in (Float32, Float64), T2 in (Float32, Float64), T3 in (Float32, Float64)
        At, xt, yt = T1.(A), T2.(x), T3.(y)
        @test B.add_outer(At, xt) ≈ At + xt * xt'
        @test B.add_outer(At, xt, yt) ≈ At + xt * yt'
        @test eltype(B.add_outer(At, xt, yt)) === promote_type(T1, T2, T3)
    end
    @test B.add_outer(A, x) !== A && A == [2.0 1.0; 0.5 3.0]     # a new matrix, `A` untouched
    @test eltype(B.add_outer(big.(A), x)) === BigFloat              # the generic loop
    @test B.add_outer(2.0, 3.0) == 11.0 && B.add_outer(2.0, 3.0, 4.0) == 14.0

    C = [1.0 2.0; -1.0 0.5]
    @test B.trace_product(A, C) ≈ tr(A * C)
    @test B.trace_product(Float32.(A), C) isa Float64
    @test B.trace_product(2.0, 3.0) == 6.0
    @test_throws DimensionMismatch B.trace_product(A, ones(3, 3))
    @test_throws DimensionMismatch B.trace_product(ones(2, 3), ones(2, 3))

    # A dense `Array` is overwritten; a view and a number are not.
    D = copy(A)
    @test B.negate!!(D) == -A && D == -A
    V = view(copy(A), :, :)
    @test B.negate!!(V) == -A && V == A
    @test B.negate!!(2.0) == -2.0
    D = copy(A)
    @test B.scale!!(3.0, D) == 3.0 * A && D == 3.0 * A
    @test B.scale!!(3, copy(A)) == 3 * A                            # another element type: a new value
    @test B.scale!!(3.0, 2.0) == 6.0

    S = B.scaled_outer(x, 2.0)
    @test S ≈ 2.0 * x * x' && issymmetric(S)
    @test B.scaled_outer(2.0, 3.0) == 12.0
    @test B.scaled_outer(x, [1.0;;]) ≈ x * x'

    @test B.diageye(Float32, 2) == Float32[1 0; 0 1] && eltype(B.diageye(Float32, 2)) === Float32
    @test B.diageye(3) == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] && eltype(B.diageye(3)) === Float64
end

@testitem "math:promote_cluster" tags = [:base] begin
    import MessagePassingRulesBase as B
    using MessagePassingRulesBase: FactorizedCluster
    using BayesBase

    cluster = FactorizedCluster((:out,) => PointMass(1.0f0), (:v,) => PointMass(2.0f0))
    promoted = B.promote_cluster(cluster, PointMass(1.0f0), PointMass(3.0))
    @test paramfloattype(promoted[(:out,)]) === Float64 && paramfloattype(promoted[(:v,)]) === Float64
    @test B.promote_cluster(cluster, PointMass(1.0f0)) == cluster
end

@testitem "math:gaussians" tags = [:base] begin
    import MessagePassingRulesBase as B
    using BayesBase, LinearAlgebra

    # A stand-in normal: its mean and covariance are all the helpers read.
    struct Moments{M, V}
        m::M
        V::V
    end
    BayesBase.mean_cov(q::Moments) = (q.m, q.V)

    m, V = [1.0, 2.0], [2.0 0.5; 0.5 1.0]
    @test B.gaussian_second_moment(Moments(m, V)) ≈ V + m * m'
    @test B.gaussian_second_moment(Moments(3.0, 0.5)) ≈ 9.5
    @test B.gaussian_second_moment(PointMass(2.0)) == 4.0
    @test B.gaussian_cross_moment([0.1 0.0; 0.2 0.3], m, [3.0, 4.0]) ≈ [0.1 0.0; 0.2 0.3] + m * [3.0, 4.0]'
    @test B.gaussian_cross_moment(0.5, 2.0, 3.0) == 6.5

    # Independent `out` and `μ`, and their joint: E[(out - μ)(out - μ)ᵀ].
    q_out, q_μ = Moments(m, V), Moments([0.0, 1.0], [1.0 0.0; 0.0 1.0])
    Δ = m - [0.0, 1.0]
    @test B.gaussian_difference_moment(q_out, q_μ) ≈ V + I + Δ * Δ'
    joint = Moments([m; [0.0, 1.0]], [V zeros(2, 2); zeros(2, 2) I])
    @test B.gaussian_difference_moment(joint) ≈ B.gaussian_difference_moment(q_out, q_μ)
    @test B.gaussian_difference_moment(Moments(1.0, 2.0), Moments(0.5, 1.0)) ≈ 3.25

    @test B.gaussian_average_energy(2, 1.0) ≈ (2 * log(2π) + 1.0) / 2
    @test B.gaussian_average_energy(2, 1.0f0) isa Float32

    # For point masses, what the factor sees is the parameter itself.
    @test B.gaussian_variational_variance(PointMass(2.0)) ≈ 2.0
    @test B.gaussian_variational_covariance(PointMass(V)) ≈ V

    @test B.gaussian_coupled_precision(1.0, 2.0, 0.5) == [1.5 -0.5; -0.5 2.5]
    W = B.gaussian_coupled_precision([1.0;;], [2.0;;], [0.5;;])
    @test W == [1.5 -0.5; -0.5 2.5]

    Λ, Λf = [2.0 0.5; 0.5 1.0], [1.0 0.0; 0.0 3.0]
    @test B.gaussian_series_precision(Λ, Λf) ≈ inv(inv(Λ) + inv(Λf))
end
