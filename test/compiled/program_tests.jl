@testitem "CompiledProgram: indexed DAG, liveness, and cycles" begin
    const R = ReactiveMP
    @test_throws ArgumentError CompiledRunner(workers = 0)
    @test_throws ArgumentError CompiledRunner(workers = Threads.nthreads() + 1)
    program = R.CompiledProgram(CompiledRunner(workers = 1))
    a = R.compiled_slot!(program, 2.0)
    b = R.compiled_slot!(program, 3.0)
    c = R.compiled_slot!(program)
    d = R.compiled_slot!(program)
    unused = R.compiled_slot!(program)
    R.compiled_operation!(program, x -> x[1] * x[2], d, (c, b))
    R.compiled_operation!(program, x -> x[1] + x[2], c, (a, b))
    R.compiled_operation!(program, x -> error("unused operation must not execute"), unused, a)
    R.prune_compiled_operations!(program, [d])
    @test length(program.operations) == 2
    R.compile_schedule!(program)
    @test length(program.order) == 2
    @test isempty(program.kernel_ids)
    @test R.compiled_csr_offsets(Int32[2, 0, 1]) == Int32[1, 3, 3, 4]
    @test eltype(R.compiled_csr_offsets([typemax(Int32)])) === Int
    R.compiled_sweep!(program)
    @test program.values[c] == 5.0
    @test program.values[d] == 15.0
    @test program.sweeps == 1
    @test !program.failed

    cycle = R.CompiledProgram(CompiledRunner(workers = 1))
    x = R.compiled_slot!(cycle, 1.0)
    y = R.compiled_slot!(cycle, 2.0)
    R.compiled_operation!(cycle, value -> value / 2, x, y)
    R.compiled_operation!(cycle, value -> value / 2, y, x)
    R.compile_schedule!(cycle)
    @test length(cycle.phase_offsets) == 3
    for _ in 1:50
        R.compiled_sweep!(cycle)
    end
    @test abs(cycle.values[x]) < 1e-20
    @test abs(cycle.values[y]) < 1e-20

    seeded = R.CompiledProgram(CompiledRunner(workers = 1))
    a = R.compiled_slot!(seeded, 1.0)
    b, c, d = R.compiled_slot!(seeded), R.compiled_slot!(seeded), R.compiled_slot!(seeded)
    for (out, input) in ((b, c), (c, d), (d, a), (a, b))
        R.compiled_operation!(seeded, x -> x / 2, out, input)
    end
    R.compile_schedule!(seeded)
    R.compiled_sweep!(seeded)
    @test all(value -> value !== nothing, seeded.values)

    # A seeded requested output must not hide an unavailable dependency.
    blocked = R.CompiledProgram(CompiledRunner(workers = 1))
    unavailable = R.compiled_slot!(blocked)
    initialized = R.compiled_slot!(blocked, 42.0)
    R.compiled_operation!(blocked, identity, initialized, unavailable)
    R.compile_schedule!(blocked)
    @test_throws "unresolved operations" R.compiled_sweep!(blocked)
    @test blocked.failed
    @test blocked.sweeps == 0

    unseeded = R.CompiledProgram(CompiledRunner(workers = 1))
    x, y = R.compiled_slot!(unseeded), R.compiled_slot!(unseeded)
    R.compiled_operation!(unseeded, identity, x, y)
    R.compiled_operation!(unseeded, identity, y, x)
    R.compile_schedule!(unseeded)
    @test_throws "unresolved operations" R.compiled_sweep!(unseeded)
    @test unseeded.failed

    failed = R.CompiledProgram(CompiledRunner(workers = 1))
    source = R.compiled_slot!(failed, 1)
    target = R.compiled_slot!(failed)
    R.compiled_operation!(failed, _ -> error("rule failed"), target, source)
    R.compile_schedule!(failed)
    @test_throws "rule failed" R.compiled_sweep!(failed)
    @test failed.failed
    @test failed.values[target] === nothing
    @test_throws ArgumentError R.compiled_sweep!(failed)
end

@testitem "CompiledProgram: typed payloads and guarded type changes" begin
    using ExponentialFamily, BayesBase
    const R = ReactiveMP
    program = R.CompiledProgram(CompiledRunner(workers = 1))
    source = R.compiled_slot!(program, Message(NormalMeanVariance(1.0, 2.0), false, false))
    output = R.compiled_slot!(program)
    kernel = input -> Message(NormalMeanPrecision(mean(input) + 1, 2.0), false, false)
    R.compiled_operation!(program, kernel, output, source)
    R.compile_schedule!(program)
    R.compiled_sweep!(program)
    expected = program.values[output]
    R.specialize_compiled_program!(program)
    @test program.values isa R.CompiledValues
    @test program.values[output] == expected
    @test length(program.typed.executors) == 1
    R.compiled_sweep!(program)
    @test program.values[output] == expected
    # Streaming/approximation updates may change a slot's distribution family.
    # The guard must use the general kernel, never assert a stale inferred type.
    program.values[source] = Message(NormalWeightedMeanPrecision(6.0, 2.0), false, false)
    R.compiled_sweep!(program)
    @test mean(program.values[output]) == 4.0
    @test iszero(program.values.tags[source])
    annotated = Message(NormalMeanPrecision(5.0, 2.0), false, false)
    R.annotate!(R.getannotations(annotated), :custom, 42)
    program.values[output] = annotated
    @test program.values[output] === annotated
    @test iszero(program.values.tags[output])
    boxed = Any[Message(PointMass([1.0, 2.0]), true, false), nothing]
    original = boxed[1]
    storage = R.pack_compiled_values(boxed)
    @test storage[1] === original
    @test !R.compiled_ready(storage, Int32(2))
end

@testitem "CompiledProgram: conflict-free phases and threaded determinism" begin
    const R = ReactiveMP
    function fixture(workers)
        program = R.CompiledProgram(CompiledRunner(; workers))
        a = R.compiled_slot!(program, 1.0)
        roots = Int32[]
        for i in 1:2048
            b, c, d = R.compiled_slot!(program), R.compiled_slot!(program), R.compiled_slot!(program)
            R.compiled_operation!(program, sum, d, (b, c))
            R.compiled_operation!(program, identity, b, a)
            R.compiled_operation!(program, x -> x * 2, c, b)
            push!(roots, d)
        end
        R.compile_schedule!(program)
        return program, roots
    end
    serial, roots = fixture(1)
    parallel, _ = fixture(Threads.nthreads())
    @test length(serial.phase_offsets) == 4
    for program in (serial, parallel)
        for phase in 1:(length(program.phase_offsets) - 1)
            ids = program.order[program.phase_offsets[phase]:(program.phase_offsets[phase + 1] - 1)]
            writes = Set(program.operations[id].output for id in ids)
            for id in ids
                R.foreach_compiled_slot(program.operations[id].inputs) do slot
                    @test slot ∉ writes
                end
            end
        end
        for _ in 1:3
            R.compiled_sweep!(program)
        end
    end
    @test serial.values == parallel.values
    @test all(slot -> parallel.values[slot] == 3.0, roots)
end
