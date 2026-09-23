@testitem "containers:messages" tags = [:base] begin
    using MessagePassingRulesBase: Messages

    m = Messages((v = 2.0, μ = 1.0))
    @test m[:μ] === 1.0
    @test m[:v] === 2.0
    @test keys(m) == (:v, :μ)
    @test_throws KeyError m[:missing]

    # Canonical order: the type does not depend on the order the caller wrote.
    @test typeof(Messages((μ = 1.0, v = 2.0))) === typeof(Messages((v = 2.0, μ = 1.0)))

    # A group is an ordinary tuple under its name.
    g = Messages((switch = 0.5, inputs = (1.0, 2.0, 3.0)))
    @test g[:inputs][2] === 2.0
    @test length(g[:inputs]) == 3

    # Underscores in interface names are ordinary characters.
    u = Messages((y_x = 1.0, y = 2.0))
    @test u[:y_x] === 1.0
    @test u[:y] === 2.0
end

@testitem "containers:marginals" tags = [:base] begin
    using MessagePassingRulesBase: Marginals

    q = Marginals((out = 1.0, p = (10.0, 20.0)), Val(((:y, :x), (:a, :b, :c))), ("joint-yx", "joint-abc"))
    @test q[:out] === 1.0
    @test q[:p][2] === 20.0
    @test q[:y, :x] === "joint-yx"
    @test q[:a, :b, :c] === "joint-abc"
    @test q[Val((:y, :x))] === "joint-yx"
    @test keys(q) == (:out, :p)

    # A joint is keyed by its ordered member tuple: (:x, :y) is a different key, and a
    # joint never collides with a single whose name contains an underscore.
    @test_throws KeyError q[:x, :y]
    u = Marginals((y_x = 1.0,), Val(((:y, :x),)), (2.0,))
    @test u[:y_x] === 1.0
    @test u[:y, :x] === 2.0

    # Canonical order for singles and for joints.
    q1 = Marginals((b = 1, a = 2), Val(((:y, :x), (:a, :b))), (3, 4))
    q2 = Marginals((a = 2, b = 1), Val(((:a, :b), (:y, :x))), (4, 3))
    @test typeof(q1) === typeof(q2)
    @test q1[:a, :b] === q2[:a, :b] === 4

    @test Marginals((out = 1.0,))[:out] === 1.0
end

@testitem "containers:args" tags = [:base] begin
    using MessagePassingRulesBase: RuleArgs, Messages, Marginals

    args = RuleArgs(m = (μ = 1.0,), q = (τ = 2.0,))
    @test args.m isa Messages
    @test args.q isa Marginals
    @test args.m[:μ] === 1.0
    @test args.q[:τ] === 2.0

    empty = RuleArgs()
    @test keys(empty.m) == ()
    @test keys(empty.q) == ()
end

@testitem "public_equivalent is the identity by default" tags = [:base] begin
    using MessagePassingRulesBase

    for d in (1.0, [1.0, 2.0], "a distribution", nothing)
        @test public_equivalent(d) === d
    end
end
