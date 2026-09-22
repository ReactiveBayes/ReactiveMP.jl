# Phase 0 -- one worked allocation example, end to end, using the `preallocate` lowering
# written by hand. (`@allocate` is deleted; `preallocate` is a keyword.)
#
#     @define_message_update_rule(
#         node        = MvNormalMeanPrecision,
#         towards     = :out,
#         inplace     = true,
#         args        = (m[:out]::MvNormalMeanPrecision, m[:μ]::MvNormalMeanPrecision),
#         preallocate = (args) -> MvNormalMeanPrecision(
#             buffer_like(mean(args.m[:μ])), buffer_like(precision(args.m[:μ])),
#         ),
#         body = (output::MvNormalMeanPrecision, args) -> begin
#             ...                       # writes straight into the buffer
#             output
#         end,
#     )
#
# Three things this has to demonstrate, from PLAN.md § In-place rules:
#   1. `rule(...)` is "resolve, preallocate, run"; `rule!(buffer, ...)` is "resolve, run",
#      and the two agree numerically;
#   2. in-place does NOT mean non-allocating -- they are separate properties, and the
#      allocating entry point obviously allocates its buffer;
#   3. the body may use `output` AND `args.m[:out]` together. The earlier design forbade
#      this, because `m[target]` had to mean either the buffer or the inbound message and
#      could not mean both. With `output` as its own slot the ambiguity is gone.

include("02_rules.jl")

using .Rules
using .Rules.Machinery
using BayesBase, ExponentialFamily, LinearAlgebra, Test, Printf

const RESULTS = String[]
record(args...) = (s = string(args...); push!(RESULTS, s); println(s))

struct Damped end   # node marker

# `buffer_like` -- dispatches on the SOURCE, rather than allocating a default and
# converting. PLAN.md § In-place rules: this is the extension seam for array types and
# devices, and the reason `preallocate` is where device placement would later belong.
buffer_like(x::Array) = similar(x)
buffer_like(x::AbstractMatrix) = similar(Array(x))
buffer_like(x::AbstractVector) = similar(Array(x))

# The rule: out = (inbound out + μ) / 2, written into a provided buffer. Contrived, but it
# uses both `output` and `args.m[:out]`, which is the point.
const SPEC_DAMPED = RuleSpec(
    (output::MvNormalMeanPrecision, algo, ctx, args, ann, node, target) -> begin
        prev = args.m[:out]
        μ    = args.m[:μ]
        om, pm = mean(output), mean(prev)
        @inbounds for i in eachindex(om)
            om[i] = (pm[i] + mean(μ)[i]) / 2
        end
        copyto!(precision(output), precision(μ))
        return output
    end;
    prealloc = (args) -> MvNormalMeanPrecision(
        buffer_like(mean(args.m[:μ])),
        buffer_like(precision(args.m[:μ])),
    ),
    inplace = true,
    source = "(output::MvNormalMeanPrecision, args) -> ...",
    file = Symbol(@__FILE__), line = @__LINE__,
)

Machinery.find_rule(::Type{Damped}, ::Target{:out}, ::BP, ::RuleArgs{<:NamedTuple{(:out, :μ)}}) = SPEC_DAMPED

mkargs() = RuleArgs(m = (
    out = MvNormalMeanPrecision([0.0, 0.0], Matrix(1.0I, 2, 2)),
    μ   = MvNormalMeanPrecision([2.0, 4.0], Matrix(2.0I, 2, 2)),
))

allocating(a) = message_passing_rule(Damped, Target(:out), BP(), a)
inplace!(buf, a) = message_passing_rule!(buf, Damped, Target(:out), BP(), a)

function main()
    record("# Phase 0 -- worked allocation example")
    record("julia = ", VERSION)
    record("")

    args = mkargs()

    r_alloc = allocating(args)
    buf     = (SPEC_DAMPED.prealloc)(args)
    r_inpl  = inplace!(buf, args)

    record("## 1. `rule` and `rule!` agree")
    record("  rule(...)  mean = ", mean(r_alloc), "  precision diag = ", diag(precision(r_alloc)))
    record("  rule!(...) mean = ", mean(r_inpl),  "  precision diag = ", diag(precision(r_inpl)))
    record("  identical: ", mean(r_alloc) == mean(r_inpl) && precision(r_alloc) == precision(r_inpl))
    record("")

    record("## 2. in-place is not the same property as non-allocating")
    a1 = (allocating(args); @allocated allocating(args))
    a2 = (inplace!(buf, args); @allocated inplace!(buf, args))
    record(@sprintf("  rule(...)          allocates %4d bytes  <- it builds the buffer, by definition", a1))
    record(@sprintf("  rule!(buffer, ...) allocates %4d bytes  <- the kernel itself", a2))
    record("")
    record("  PLAN.md § Verification asks for three distinct levels, never conflated:")
    record("  kernel (== 0), rule with a provided buffer (== 0), full sweep (golden + tol).")
    record("  The first two are the two numbers above; the third belongs to the engine.")
    record("")

    record("## 3. `output` and `args.m[:out]` coexist")
    record("  the body reads the inbound message on its own edge while writing the buffer;")
    record("  result depends on both: mean(prev)=", mean(args.m[:out]), " mean(μ)=", mean(args.m[:μ]),
           " -> ", mean(r_inpl))
    record("")

    record("## 4. The output type is pinned by the lambda parameter, not by macro analysis")
    wrong = MvNormalMeanPrecision([0.0], Matrix(1.0I, 1, 1))
    pinned = try
        SPEC_DAMPED.body("not a distribution", BP(), default_context, args, NoAnnotations(), Damped, Target(:out))
        false
    catch err
        record("  passing a wrong `output` gives ", typeof(err), " -- Julia's own check")
        err isa MethodError
    end
    record("")

    @testset "allocation example" begin
        @test mean(r_alloc) == mean(r_inpl)
        @test precision(r_alloc) == precision(r_inpl)
        @test mean(r_inpl) == [1.0, 2.0]
        @test a2 == 0            # the kernel with a provided buffer allocates nothing
        @test a1 > 0             # the allocating form does allocate -- that is the point
        @test pinned
    end

    open(joinpath(@__DIR__, "..", "results", "allocate-julia-$(VERSION).txt"), "w") do io
        println(io, join(RESULTS, "\n"))
    end
    return nothing
end

main()
