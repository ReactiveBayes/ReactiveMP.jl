@testmodule InplaceRules begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: buffer_like
    using LinearAlgebra: mul!

    struct Affine end
    @define_factor_node(node = Affine, type = Deterministic, interfaces = [:out, :A, :x])
    @define_message_update_rule(
        node = Affine,
        target = :out,
        inplace = true,
        args = (m[:A]::Matrix{Float64}, m[:x]::Vector{Float64}),
        preallocate = (args) -> buffer_like(args.m[:x]),
        body = (output::Vector{Float64}, args) -> mul!(output, args.m[:A], args.m[:x]),
    )

    using MessagePassingRulesBase: RuleArgs, Target, DefaultAlgorithm
    const ARGS = RuleArgs(m = (A = [1.0 2.0; 3.0 4.0], x = [1.0, 1.0]))
    const BUFFER = zeros(2)
    into_buffer() = getresult(message_passing_rule!(BUFFER, Affine, Target(:out), DefaultAlgorithm(), ARGS))
    allocating() = getresult(message_passing_rule(Affine, Target(:out), DefaultAlgorithm(), ARGS))
    measure(f) = (f(); @allocated f())
end

@testitem "inplace:agreement" tags = [:base] setup = [InplaceRules] begin
    I = InplaceRules
    @test I.allocating() == [3.0, 7.0]
    @test I.into_buffer() === I.BUFFER
    @test I.BUFFER == I.allocating()
end

@testitem "inplace:allocations" tags = [:base, :alloc] setup = [InplaceRules] begin
    # In-place with a provided buffer allocates nothing; the allocating form pays for its
    # buffer, which is the negative control.
    I = InplaceRules
    @test I.measure(I.into_buffer) == 0
    @test I.measure(I.allocating) > 0
end

@testitem "inplace:buffer_like" tags = [:base] begin
    using MessagePassingRulesBase: buffer_like
    v = [1.0, 2.0]
    b = buffer_like(v)
    @test b isa Vector{Float64} && size(b) == size(v) && b !== v
    @test buffer_like(v, Float32) isa Vector{Float32}
    m = buffer_like([1 2; 3 4])
    @test m isa Matrix{Int} && size(m) == (2, 2)
    nested = buffer_like((a = [1.0], b = ([2.0, 3.0],)))
    @test nested.a isa Vector{Float64} && nested.b[1] isa Vector{Float64} && length(nested.b[1]) == 2
    @test_throws ArgumentError buffer_like(1.0)
end
