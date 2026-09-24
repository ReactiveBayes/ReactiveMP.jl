# An affine f = B + reshape(a) is the linear model on y - B x: under q(y, x), every rule and the
# energy for it equal those for reshape on the joint of (y - B x, x). v6's rules towards `a` and
# `W` took the rows of A as linear through the origin and dropped B; they keep it now.

@testitem "rules:ContinuousTransition:an affine transformation is the linear one on y - B x" tags = [:rules] begin
    using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra

    spd(d, s) = [i == j ? s + i : 0.2 / (i + j) for i in 1:d, j in 1:d]
    close(p, q) = isapprox(mean(p), mean(q); atol = 1.0e-9) && isapprox(cov(p), cov(q); atol = 1.0e-9)

    for (dy, dx) in ((2, 2), (1, 2), (2, 3))
        B = [0.5 * i - 0.3 * j for i in 1:dy, j in 1:dx]
        affine, linear = CTVMP(a -> B + reshape(a, dy, dx)), CTVMP(a -> reshape(a, dy, dx))
        q_a = MvNormalMeanCovariance([0.3 - 0.7 * (k - 1) / max(dy * dx - 1, 1) for k in 1:(dy * dx)], 0.05 * spd(dy * dx, 1.0))
        q_W = Wishart(dy + 3, spd(dy, 0.5) / 3)
        q_y_x = MvNormalMeanCovariance([1.0 - 0.3 * k for k in 1:(dy + dx)], 0.3 * spd(dy + dx, 1.5))
        # (y - B x, x) = T (y, x)
        T = [Matrix(1.0I, dy, dy) -B; zeros(dx, dy) Matrix(1.0I, dx, dx)]
        shifted = MvNormalMeanCovariance(T * mean(q_y_x), T * cov(q_y_x) * T')

        @test close(
            call_message_update_rule(ContinuousTransition, :a; clusters = ((:y, :x) => q_y_x,), q = (a = q_a, W = q_W), algorithm = affine),
            call_message_update_rule(ContinuousTransition, :a; clusters = ((:y, :x) => shifted,), q = (a = q_a, W = q_W), algorithm = linear),
        )
        W_affine = call_message_update_rule(ContinuousTransition, :W; clusters = ((:y, :x) => q_y_x,), q = (a = q_a,), algorithm = affine)
        W_linear = call_message_update_rule(ContinuousTransition, :W; clusters = ((:y, :x) => shifted,), q = (a = q_a,), algorithm = linear)
        @test all(map((p, q) -> isapprox(p, q; atol = 1.0e-9), params(W_affine), params(W_linear)))
        @test call_average_energy(ContinuousTransition; clusters = ((:y, :x) => q_y_x,), q = (a = q_a, W = q_W), algorithm = affine) ≈
            call_average_energy(ContinuousTransition; clusters = ((:y, :x) => shifted,), q = (a = q_a, W = q_W), algorithm = linear)

        # Under mean-field, the offset reaches `a` through E[x xᵀ]: the message equals the one for
        # the linear model on the joint (y - B x, x), whose cross-covariance is -B Vx.
        q_y, q_x = MvNormalMeanCovariance(mean(q_y_x)[1:dy], cov(q_y_x)[1:dy, 1:dy]), MvNormalMeanCovariance(mean(q_y_x)[(dy + 1):end], cov(q_y_x)[(dy + 1):end, (dy + 1):end])
        independent = MvNormalMeanCovariance(vcat(mean(q_y), mean(q_x)), [cov(q_y) zeros(dy, dx); zeros(dx, dy) cov(q_x)])
        independent_shifted = MvNormalMeanCovariance(T * mean(independent), T * cov(independent) * T')
        @test close(
            call_message_update_rule(ContinuousTransition, :a; q = (y = q_y, x = q_x, a = q_a, W = q_W), algorithm = affine),
            call_message_update_rule(ContinuousTransition, :a; clusters = ((:y, :x) => independent_shifted,), q = (a = q_a, W = q_W), algorithm = linear),
        )
        W_meanfield = call_message_update_rule(ContinuousTransition, :W; q = (y = q_y, x = q_x, a = q_a), algorithm = affine)
        W_shifted = call_message_update_rule(ContinuousTransition, :W; clusters = ((:y, :x) => independent_shifted,), q = (a = q_a,), algorithm = linear)
        @test all(map((p, q) -> isapprox(p, q; atol = 1.0e-9), params(W_meanfield), params(W_shifted)))
    end
end
