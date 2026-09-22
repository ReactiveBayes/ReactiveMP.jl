# The Phase 3 step 2 gate: container access inside a real lambda body resolves at compile
# time. Each access is measured through a function barrier with `const` bodies, never at
# global scope (see DISCUSSION.md §3.15 for how each of those mistakes skews the number).

@testmodule GateBodies begin
    using MessagePassingRulesBase: RuleArgs, Marginals

    const single = (args) -> args.m[:μ]
    const group = (args) -> args.q[:p][2]
    const joint2 = (args) -> args.q[:y, :x]
    const joint3 = (args) -> args.q[:a, :b, :c]
    const mixed = (args) -> args.m[:μ] + args.q[:p][1] + args.q[:y, :x]

    # Negative control: the key is a runtime value, so nothing can be resolved statically.
    const dynamic = (args, a, b) -> args.q[a, b]

    function make_args()
        return RuleArgs(
            m = (μ = 1.0, v = 2.0),
            q = Marginals((out = 1.0, p = (3.0, 4.0)), Val(((:y, :x), (:a, :b, :c))), (5.0, 6.0)),
        )
    end

    measure(body, args) = (body(args); @allocated body(args))
end

@testitem "gate:containers" tags = [:base, :alloc] setup = [GateBodies] begin
    args = GateBodies.make_args()
    for body in (GateBodies.single, GateBodies.group, GateBodies.joint2, GateBodies.joint3, GateBodies.mixed)
        @test (@inferred body(args)) isa Float64
        @test GateBodies.measure(body, args) == 0
    end
    @test GateBodies.mixed(args) == 1.0 + 3.0 + 5.0
end

@testitem "gate:containers-negative-control" tags = [:base, :alloc] setup = [GateBodies] begin
    # If this ever infers concretely, the gate above has stopped measuring anything.
    args = GateBodies.make_args()
    T = only(Base.return_types(GateBodies.dynamic, (typeof(args), Symbol, Symbol)))
    @test !isconcretetype(T)
    @test GateBodies.dynamic(args, :y, :x) == 5.0
end

@testitem "gate:containers-jet" tags = [:base] setup = [GateBodies] begin
    using JET
    args = GateBodies.make_args()
    for body in (GateBodies.single, GateBodies.group, GateBodies.joint2, GateBodies.joint3, GateBodies.mixed)
        JET.@test_opt body(args)
    end
end
