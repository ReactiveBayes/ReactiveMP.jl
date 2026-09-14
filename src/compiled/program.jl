export CompiledRunner, UnsupportedCompiledFeature

"""
    CompiledRunner(; workers = Threads.nthreads())

Opt in to the experimental compact, explicitly scheduled inference backend.
This is not a stream postprocessor. Unsupported extensions raise an explicit
error; this backend never falls back to reactive inference.
"""
struct CompiledRunner
    workers::Int
    optimize::Bool
    function CompiledRunner(; workers::Integer = Threads.nthreads(), optimize::Bool = true)
        1 <= workers <= Threads.nthreads() || throw(ArgumentError("workers must be between 1 and Threads.nthreads()"))
        return new(workers, optimize)
    end
end

struct UnsupportedCompiledFeature <: Exception
    feature::String
end
Base.showerror(io::IO, error::UnsupportedCompiledFeature) = print(io, "CompiledRunner does not support ", error.feature, ". No reactive fallback was used.")

"A variable identity usable by product rules, without observable fields."
struct CompiledVariable <: AbstractVariable
    id::Int32
    label::Any
    degree::Int32
    kind::UInt8 # 0=random, 1=data, 2=constant
end
degree(variable::CompiledVariable) = Int(variable.degree)
israndom(variable::CompiledVariable) = variable.kind == 0
isdata(variable::CompiledVariable) = variable.kind == 1
isconst(variable::CompiledVariable) = variable.kind == 2

"Lightweight identities for existing declarative dependency-policy methods."
struct CompiledInterface
    name::Symbol
    variable::CompiledVariable
    message::Int32
    marginal::Int32
end
name(interface::CompiledInterface) = interface.name
tag(interface::CompiledInterface) = Val(interface.name)
getvariable(interface::CompiledInterface) = interface.variable
struct CompiledLocalMarginal
    name::Symbol
    slot::Int32
end
name(marginal::CompiledLocalMarginal) = marginal.name
tag(marginal::CompiledLocalMarginal) = Val(marginal.name)
get_stream_of_inbound_messages(::CompiledInterface) = throw(UnsupportedCompiledFeature("a dependency policy that requires reactive streams"))
get_stream_of_marginals(::CompiledLocalMarginal) = throw(UnsupportedCompiledFeature("a dependency policy that requires reactive streams"))

struct CompiledManyOf{T}
    slots::T
end

"Shared input list with one excluded edge; avoids quadratic IR on hub variables."
struct CompiledExcept{S <: AbstractVector{Int32}}
    slots::S
    excluded::Int
end

struct CompiledMessageView{V, B <: CompiledExcept} <: AbstractVector{Any}
    values::V
    binding::B
end
Base.size(view::CompiledMessageView) = (length(view.binding.slots) - 1,)
Base.IndexStyle(::Type{<:CompiledMessageView}) = IndexLinear()
function Base.getindex(view::CompiledMessageView, i::Int)
    @boundscheck checkbounds(view, i)
    j = i < view.binding.excluded ? i : i + 1
    return view.values[view.binding.slots[j]]
end

compiled_read(values, slot::Int32) = values[slot]
compiled_read(values, ::Nothing) = nothing
compiled_read(values, slots::Tuple) = map(slot -> compiled_read(values, slot), slots)
compiled_read(values, slots::AbstractVector{Int32}) = map(slot -> compiled_read(values, slot), slots)
compiled_read(values, slots::CompiledManyOf) = ManyOf(compiled_read(values, slots.slots))
compiled_read(values, slots::CompiledExcept) = CompiledMessageView(values, slots)
compiled_ready(values, slot::Int32) = values[slot] !== nothing
compiled_ready(values, ::Nothing) = true
compiled_ready(values, slots::Tuple) = all(slot -> compiled_ready(values, slot), slots)
compiled_ready(values, slots::AbstractVector{Int32}) = all(slot -> compiled_ready(values, slot), slots)
compiled_ready(values, slots::CompiledManyOf) = compiled_ready(values, slots.slots)
compiled_ready(values, slots::CompiledExcept) = all(i -> i == slots.excluded || compiled_ready(values, slots.slots[i]), eachindex(slots.slots))

foreach_compiled_slot(f, slot::Int32) = f(slot)
foreach_compiled_slot(f, ::Nothing) = nothing
foreach_compiled_slot(f, slots::Tuple) = foreach(slot -> foreach_compiled_slot(f, slot), slots)
foreach_compiled_slot(f, slots::AbstractVector{Int32}) = foreach(f, slots)
foreach_compiled_slot(f, slots::CompiledManyOf) = foreach_compiled_slot(f, slots.slots)
function foreach_compiled_slot(f, slots::CompiledExcept)
    for i in eachindex(slots.slots)
        i == slots.excluded || f(slots.slots[i])
    end
end

struct CompiledMessageKernel{M}
    mapping::M
end
(kernel::CompiledMessageKernel)(inputs) = kernel.mapping(inputs[1], inputs[2])

struct CompiledMarginalKernel{M}
    mapping::M
end
(kernel::CompiledMarginalKernel)(inputs) = compute_marginal(kernel.mapping, inputs[1], inputs[2])

struct CompiledProductKernel{C}
    variable::CompiledVariable
    context::C
    marginal::Bool
end
function (kernel::CompiledProductKernel)(inputs)
    result = compute_product_of_messages(kernel.variable, kernel.context, inputs)
    return kernel.marginal ? as_marginal(result) : result
end

struct CompiledAsMarginalKernel end
(::CompiledAsMarginalKernel)(input) = as_marginal(input)

"A coordinate update: evaluate single-use VMP messages immediately before their product."
struct CompiledVariationalProductKernel{K}
    product::K
    messages::Vector{Any} # nothing means an already-computed, unfused message
end
function (kernel::CompiledVariationalProductKernel)(inputs)
    messages = map(eachindex(kernel.messages)) do i
        message = kernel.messages[i]
        message === nothing ? inputs[i] : message(inputs[i])
    end
    return kernel.product(messages)
end

# Unknown extensions can opt in with a method after auditing shared state.
# Stateful approximations and callbacks still use this backend, but serially.
compiled_parallel_safe(::Any) = false
compiled_callbacks_parallel_safe(::Any) = false
compiled_callbacks_parallel_safe(::Nothing) = true
function compiled_parallel_safe(kernel::CompiledMessageKernel)
    mapping = kernel.mapping
    return compiled_callbacks_parallel_safe(mapping.callbacks) && mapping.annotations === nothing &&
        mapping.rulefallback === nothing && mapping.factornode === nothing &&
        multicore_readonly(mapping.meta) && multicore_readonly(mapping.vconstraint)
end
compiled_parallel_safe(kernel::CompiledMarginalKernel) = multicore_readonly(kernel.mapping)
function compiled_parallel_safe(kernel::CompiledProductKernel)
    context = kernel.context
    return compiled_callbacks_parallel_safe(context.callbacks) && context.annotations === nothing &&
        all(field -> field === :callbacks || multicore_readonly(getfield(context, field)), fieldnames(typeof(context)))
end
compiled_parallel_safe(::CompiledAsMarginalKernel) = true
compiled_parallel_safe(kernel::CompiledVariationalProductKernel) =
    compiled_parallel_safe(kernel.product) && all(message -> message === nothing || compiled_parallel_safe(message), kernel.messages)

struct CompiledLinkKernel{F, A}
    transform::F
    arguments::A
end
struct CompiledLiteral{T}
    value::T
end
compiled_parallel_safe(kernel::CompiledLinkKernel) = multicore_readonly(kernel.transform)
compiled_link_arg(arg::CompiledLiteral, inputs) = arg.value
compiled_link_arg(index::Int, inputs) = mean(getdata(inputs[index]))
compiled_link_arg(args::AbstractArray, inputs) = map(arg -> compiled_link_arg(arg, inputs), args)
function (kernel::CompiledLinkKernel)(inputs)
    any(input -> ismissing(getdata(input)), inputs) && return Message(missing, false, false)
    all(input -> getdata(input) isa PointMass, inputs) || throw(ArgumentError("Linked data requires PointMass inputs"))
    args = map(arg -> compiled_link_arg(arg, inputs), kernel.arguments)
    return Message(PointMass(kernel.transform(args...)), false, false)
end

struct CompiledOperation
    kernel::Int32
    output::Int32
    inputs::Any
end

mutable struct CompiledProgram
    config::CompiledRunner
    kernels::Vector{Any}
    kernel_ids::Dict{Any, Int32}
    operations::Vector{CompiledOperation}
    values::AbstractVector{Any}
    order::Vector{Int32}
    phase_offsets::Vector{Int}
    phase_parallel::BitVector
    typed::Any
    sweeps::Int
    failed::Bool
end

CompiledProgram(config::CompiledRunner) = CompiledProgram(config, Any[], Dict{Any, Int32}(), CompiledOperation[], Any[], Int32[], Int[1], BitVector(), nothing, 0, false)

function compiled_slot!(program::CompiledProgram, value = nothing)
    length(program.values) < typemax(Int32) || throw(OverflowError("Compiled message slot capacity exceeded"))
    push!(program.values, value)
    return Int32(length(program.values))
end

function compiled_operation!(program::CompiledProgram, kernel, output::Int32, inputs)
    kid = get!(program.kernel_ids, kernel) do
        push!(program.kernels, kernel)
        Int32(length(program.kernels))
    end
    push!(program.operations, CompiledOperation(kid, output, inputs))
    return output
end

function compiled_producers(program::CompiledProgram)
    producers = zeros(Int32, length(program.values))
    for (index, operation) in enumerate(program.operations)
        iszero(producers[operation.output]) || error("Multiple writers for compiled slot $(operation.output)")
        producers[operation.output] = index
    end
    return producers
end

"Remove operations not needed by requested posteriors, dependencies, or scores."
function prune_compiled_operations!(program::CompiledProgram, roots)
    producers = compiled_producers(program)
    live = falses(length(program.operations))
    stack = Int32[]
    function visit(slot)
        producer = producers[slot]
        if !iszero(producer) && !live[producer]
            live[producer] = true
            push!(stack, producer)
        end
    end
    foreach(visit, roots)
    while !isempty(stack)
        index = pop!(stack)
        foreach_compiled_slot(visit, program.operations[index].inputs)
    end
    keepat!(program.operations, live)
    # Drop dead kernels as well as dead operations. In particular, an unused
    # prediction product can otherwise retain variable/context metadata.
    remap = zeros(Int32, length(program.kernels))
    kernels = Any[]
    for i in eachindex(program.operations)
        operation = program.operations[i]
        if iszero(remap[operation.kernel])
            push!(kernels, program.kernels[operation.kernel])
            remap[operation.kernel] = length(kernels)
        end
        program.operations[i] = CompiledOperation(remap[operation.kernel], operation.output, operation.inputs)
    end
    program.kernels = kernels
    program.kernel_ids = Dict{Any, Int32}()
    # Dead initialization messages may be large even when their producer never
    # ran. Keep public slot IDs stable, but release values with no live reader.
    live_values = falses(length(program.values))
    foreach(slot -> live_values[slot] = true, roots)
    for operation in program.operations
        live_values[operation.output] = true
        foreach_compiled_slot(slot -> live_values[slot] = true, operation.inputs)
    end
    for i in eachindex(program.values)
        live_values[i] || (program.values[i] = nothing)
    end
    return program
end

compiled_variational_message(::Any) = false
compiled_variational_message(kernel::CompiledMessageKernel) =
    kernel.mapping.msgs_names === nothing && kernel.mapping.marginals_names !== nothing

"""
Fuse single-consumer variational messages into their marginal product. Coloring
these coordinate updates exposes the actual q-to-q dependencies; coloring the
unfused bipartite message/product graph would instead make all parameters update
from stale marginals (a Jacobi iteration, which can oscillate even for VMP).
Multi-consumer and cavity messages retain their original operations and slots.
"""
function fuse_compiled_variational_products!(program::CompiledProgram, roots)
    candidates = map(compiled_variational_message, program.kernels)
    any(candidates) || return program
    producers = compiled_producers(program)
    readers = zeros(Int32, length(program.values))
    foreach(slot -> readers[slot] += 1, roots)
    for operation in program.operations
        foreach_compiled_slot(slot -> readers[slot] += 1, operation.inputs)
    end
    removed = falses(length(program.operations))
    for i in eachindex(program.operations)
        operation = program.operations[i]
        product = program.kernels[operation.kernel]
        product isa CompiledProductKernel && product.marginal || continue
        operation.inputs isa Union{Tuple, AbstractVector{Int32}} || continue
        messages, inputs = Any[], Any[]
        fused = false
        for slot in operation.inputs
            producer = producers[slot]
            if readers[slot] == 1 && !iszero(producer) &&
                program.operations[producer].kernel <= length(candidates) && candidates[program.operations[producer].kernel]
                message = program.operations[producer]
                push!(messages, program.kernels[message.kernel])
                push!(inputs, message.inputs)
                removed[producer] = true
                fused = true
            else
                push!(messages, nothing)
                push!(inputs, slot)
            end
        end
        if fused
            push!(program.kernels, CompiledVariationalProductKernel(product, messages))
            program.operations[i] = CompiledOperation(Int32(length(program.kernels)), operation.output, Tuple(inputs))
        end
    end
    keepat!(program.operations, .!removed)
    return prune_compiled_operations!(program, roots)
end

"Build operation dependencies and their transpose in flat CSR storage."
function compiled_dependency_csr(program::CompiledProgram)
    producers = compiled_producers(program)
    n = length(program.operations)
    counts = zeros(Int32, n)
    reverse_counts = zeros(Int32, n)
    for (index, operation) in enumerate(program.operations)
        foreach_compiled_slot(operation.inputs) do slot
            producer = producers[slot]
            if !iszero(producer)
                counts[index] = Base.checked_add(counts[index], Int32(1))
                reverse_counts[producer] = Base.checked_add(reverse_counts[producer], Int32(1))
            end
        end
    end
    offsets = compiled_csr_offsets(counts)
    reverse_offsets = compiled_csr_offsets(reverse_counts)
    dependencies = Vector{Int32}(undef, offsets[end] - 1)
    for (index, operation) in enumerate(program.operations)
        position = Int(offsets[index])
        foreach_compiled_slot(operation.inputs) do slot
            producer = producers[slot]
            if !iszero(producer)
                dependencies[position] = producer
                position += 1
            end
        end
    end
    positions = reverse_offsets[1:end-1]
    consumers = Vector{Int32}(undef, length(dependencies))
    for i in 1:n, edge in offsets[i]:(offsets[i + 1] - 1)
        producer = dependencies[edge]
        consumers[positions[producer]] = i
        positions[producer] += 1
    end
    return offsets, dependencies, reverse_offsets, consumers
end

"Exact CSR allocation; use 32-bit offsets when the edge count permits it."
function compiled_csr_offsets(counts)
    total = sum(Int, counts)
    T = total < typemax(Int32) ? Int32 : Int
    offsets = Vector{T}(undef, length(counts) + 1)
    offsets[1] = 1
    for i in eachindex(counts)
        offsets[i + 1] = offsets[i] + counts[i]
    end
    return offsets
end

"Iterative Kosaraju traversal: no Julia recursion proportional to graph size."
function compiled_components(offsets, edges, reverse_offsets, reverse_edges)
    n = length(offsets) - 1
    seen = falses(n)
    finish = Int32[]
    stack = Int32[]
    cursor = eltype(offsets)[]
    for root in 1:n
        seen[root] && continue
        seen[root] = true
        push!(stack, root)
        push!(cursor, offsets[root])
        while !isempty(stack)
            node = stack[end]
            edge = cursor[end]
            if edge < offsets[node + 1]
                cursor[end] += 1
                next = edges[edge]
                if !seen[next]
                    seen[next] = true
                    push!(stack, next)
                    push!(cursor, offsets[next])
                end
            else
                push!(finish, pop!(stack))
                pop!(cursor)
            end
        end
    end
    component = zeros(Int32, n)
    count = 0
    for root in Iterators.reverse(finish)
        !iszero(component[root]) && continue
        count += 1
        component[root] = count
        push!(stack, root)
        while !isempty(stack)
            node = pop!(stack)
            for edge in reverse_offsets[node]:(reverse_offsets[node + 1] - 1)
                next = reverse_edges[edge]
                if iszero(component[next])
                    component[next] = count
                    push!(stack, next)
                end
            end
        end
    end
    return component, count
end

function compile_schedule!(program::CompiledProgram)
    deps_offsets, deps, users_offsets, users = compiled_dependency_csr(program)
    @debug "Compiled schedule: dependencies complete" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    # Edges producer -> consumer give topologically ordered component IDs.
    components, ncomponents = compiled_components(users_offsets, users, deps_offsets, deps)
    @debug "Compiled schedule: components complete" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    n = length(program.operations)
    counts = zeros(Int32, ncomponents)
    foreach(c -> counts[c] += 1, components)
    offsets = compiled_csr_offsets(counts)
    positions = offsets[1:end-1]
    members = Vector{Int32}(undef, n)
    for i in 1:n
        c = components[i]
        members[positions[c]] = i
        positions[c] += 1
    end
    colors = zeros(Int32, n)
    phases = zeros(Int32, n)
    component_end = zeros(Int32, ncomponents)
    marks = Int32[]
    for c in 1:ncomponents
        first, last = offsets[c], offsets[c + 1] - 1
        # Independent components share phases. A component starts only after
        # every external producer has finished, including cyclic components.
        base = 0
        for p in first:last
            i = members[p]
            for edge in deps_offsets[i]:(deps_offsets[i + 1] - 1)
                producer = deps[edge]
                components[producer] == c && continue
                base = max(base, component_end[components[producer]])
            end
        end
        if first == last
            phases[members[first]] = base + 1
            component_end[c] = base + 1
            continue
        end
        maxcolor = 0
        for p in first:last
            i = members[p]
            for (adjoffsets, adjacent) in ((deps_offsets, deps), (users_offsets, users))
                for edge in adjoffsets[i]:(adjoffsets[i + 1] - 1)
                    neighbor = adjacent[edge]
                    color = colors[neighbor]
                    if components[neighbor] == c && color > 0
                        marks[color] = i
                    end
                end
            end
            color = 1
            while color <= length(marks) && marks[color] == i
                color += 1
            end
            if color > length(marks)
                push!(marks, 0)
            end
            colors[i] = color
            phases[i] = base + color
            maxcolor = max(maxcolor, color)
        end
        component_end[c] = base + maxcolor
    end
    nphases = maximum(phases; init = 0)
    phase_counts = zeros(Int, nphases)
    foreach(phase -> phase_counts[phase] += 1, phases)
    program.phase_offsets = vcat(1, 1 .+ cumsum(phase_counts))
    phase_positions = copy(program.phase_offsets[1:end-1])
    resize!(program.order, n)
    program.phase_parallel = trues(nphases)
    kernel_safe = map(compiled_parallel_safe, program.kernels)
    for i in 1:n
        phase = phases[i]
        program.order[phase_positions[phase]] = i
        phase_positions[phase] += 1
        program.phase_parallel[phase] &= kernel_safe[program.operations[i].kernel]
    end
    program.kernel_ids = Dict{Any, Int32}() # release hash-table capacity, not just entries
    return program
end

@noinline function execute_compiled_kernel(kernel, values, bindings)
    inputs = compiled_read(values, bindings)
    return kernel(inputs)
end

function execute_compiled_operation!(program::CompiledProgram, index::Int32)
    operation = program.operations[index]
    compiled_ready(program.values, operation.inputs) || return false
    if program.typed !== nothing && execute_compiled_typed!(program.typed, program, index)
        return true
    end
    program.values[operation.output] = execute_compiled_kernel(program.kernels[operation.kernel], program.values, operation.inputs)
    return true
end

"Propagate startup availability without recomputing already-produced messages."
function compiled_complete_initialization!(program::CompiledProgram)
    # Initial marginals are seeds, not proof that their inference dependencies
    # can be resolved. Include blocked operations even when their output was
    # initialized, otherwise a disconnected cycle can silently return a prior.
    missing = Int32[index for index in program.order if
        program.values[program.operations[index].output] === nothing ||
        !compiled_ready(program.values, program.operations[index].inputs)]
    isempty(missing) && return program
    # Only unavailable inputs need watchers. Every waiting edge is visited once;
    # long seeded cycles must not require repeated scans of the entire graph.
    waiting = Dict{Int32, Vector{Int32}}()
    counts = zeros(Int, length(missing))
    queue = Int32[]
    for (i, index) in enumerate(missing)
        foreach_compiled_slot(program.operations[index].inputs) do slot
            if program.values[slot] === nothing
                push!(get!(Vector{Int32}, waiting, slot), i)
                counts[i] += 1
            end
        end
        iszero(counts[i]) && push!(queue, i)
    end
    cursor = 1
    while cursor <= length(queue)
        i = queue[cursor]
        cursor += 1
        index = missing[i]
        execute_compiled_operation!(program, index)
        output = program.operations[index].output
        program.values[output] === nothing && continue
        for user in get(waiting, output, ())
            counts[user] -= 1
            iszero(counts[user]) && push!(queue, user)
        end
    end
    unresolved = count(eachindex(missing)) do i
        !iszero(counts[i]) || program.values[program.operations[missing[i]].output] === nothing
    end
    iszero(unresolved) || throw(ArgumentError("Compiled inference has $unresolved unresolved operations; provide the required data or initialization"))
    return program
end

function compiled_sweep!(program::CompiledProgram)
    program.failed && throw(ArgumentError("Cannot reuse a failed compiled program; create a fresh inference instance"))
    try
        for phase in 1:(length(program.phase_offsets) - 1)
            first, last = program.phase_offsets[phase], program.phase_offsets[phase + 1] - 1
            # Coarse chunks amortize task creation for cheap scalar rules.
            # No operation in a phase reads a slot another operation writes.
            workers = program.phase_parallel[phase] ? min(program.config.workers, (last - first + 1) ÷ 256) : 1
            if workers <= 1
                for p in first:last
                    execute_compiled_operation!(program, program.order[p])
                end
            else
                @sync for worker in 1:workers
                    lo = first + (last - first + 1) * (worker - 1) ÷ workers
                    hi = first + (last - first + 1) * worker ÷ workers - 1
                    Threads.@spawn for p in lo:hi
                        execute_compiled_operation!(program, program.order[p])
                    end
                end
            end
        end
        iszero(program.sweeps) && compiled_complete_initialization!(program)
        program.sweeps += 1
        if program.config.optimize && program.sweeps == 2 && length(program.operations) >= 1024 && all(program.phase_parallel)
            specialize_compiled_program!(program)
        end
    catch
        program.failed = true
        rethrow()
    end
    return program
end
