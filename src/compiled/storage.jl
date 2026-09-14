# General typed storage: no factor- or distribution-specific numerical rules.
# Isbits distribution payloads are stored densely. Other payloads retain their
# original boxed representation; type changes safely deoptimize affected slots.
struct CompiledPackedPool{M, D}
    data::Vector{D}
    flags::Vector{UInt8}
end
CompiledPackedPool(::Type{M}, ::Type{D}, n) where {M, D} = CompiledPackedPool{M, D}(Vector{D}(undef, n), zeros(UInt8, n))

struct CompiledValues <: AbstractVector{Any}
    boxed::Vector{Any}
    tags::Vector{UInt16}
    offsets::Vector{Int32}
    pools::Vector{Any}
end
Base.size(values::CompiledValues) = size(values.boxed)
Base.IndexStyle(::Type{CompiledValues}) = IndexLinear()
@inline function compiled_unpack(pool::CompiledPackedPool{M}, i) where {M <: Message}
    flags = pool.flags[i]
    return Message(pool.data[i], !iszero(flags & 0x01), !iszero(flags & 0x02))
end
@inline function compiled_unpack(pool::CompiledPackedPool{M}, i) where {M <: Marginal}
    flags = pool.flags[i]
    return Marginal(pool.data[i], !iszero(flags & 0x01), !iszero(flags & 0x02))
end
function Base.getindex(values::CompiledValues, i::Int)
    tag = values.tags[i]
    return iszero(tag) ? values.boxed[i] : compiled_unpack(values.pools[tag], values.offsets[i])
end
@inline function compiled_pack!(pool::CompiledPackedPool{M}, i, value) where {M}
    if value isa M && isempty(getannotations(value))
        pool.data[i] = getdata(value)
        pool.flags[i] = UInt8(is_clamped(value)) | (UInt8(is_initial(value)) << 1)
        return true
    end
    return false
end
function Base.setindex!(values::CompiledValues, value, i::Int)
    tag = values.tags[i]
    if iszero(tag) || !compiled_pack!(values.pools[tag], values.offsets[i], value)
        values.tags[i] = 0
        values.boxed[i] = value
    end
    return value
end
compiled_ready(values::CompiledValues, slot::Int32) = !iszero(values.tags[slot]) || values.boxed[slot] !== nothing

compiled_packable(::Any) = false
compiled_packable(value::Union{Message{D}, Marginal{D}}) where {D} = isbitstype(D) && isempty(getannotations(value))

function pack_compiled_values(boxed::Vector{Any})
    counts = Dict{DataType, Int}()
    for value in boxed
        compiled_packable(value) || continue
        T = typeof(value)
        counts[T] = get(counts, T, 0) + 1
    end
    length(counts) < typemax(UInt16) || return boxed
    types = Dict{DataType, UInt16}()
    pools = Any[]
    for (M, count) in counts
        push!(pools, CompiledPackedPool(M, M.parameters[1], count))
        types[M] = length(pools)
    end
    storage = CompiledValues(boxed, zeros(UInt16, length(boxed)), zeros(Int32, length(boxed)), pools)
    positions = zeros(Int32, length(pools))
    for i in eachindex(boxed)
        value = boxed[i]
        compiled_packable(value) || continue
        tag = types[typeof(value)]
        positions[tag] += 1
        offset = positions[tag]
        compiled_pack!(pools[tag], offset, value)
        storage.tags[i], storage.offsets[i] = tag, offset
        boxed[i] = nothing
    end
    return storage
end

struct CompiledPackedAccess{P}
    pool::UInt16
end
struct CompiledBoxedAccess{T} end
struct CompiledManyAccess{A}
    inner::A
end
struct CompiledUnspecialized end

function compiled_accessor(values::CompiledValues, slot::Int32)
    tag = values.tags[slot]
    return iszero(tag) ? CompiledBoxedAccess{typeof(values.boxed[slot])}() : CompiledPackedAccess{typeof(values.pools[tag])}(tag)
end
compiled_accessor(values, ::Nothing) = nothing
compiled_accessor(values, slots::Tuple) = map(slot -> compiled_accessor(values, slot), slots)
compiled_accessor(values, slots::CompiledManyOf) = CompiledManyAccess(compiled_accessor(values, slots.slots))
compiled_accessor(values, _) = CompiledUnspecialized()
compiled_specializable(::CompiledUnspecialized) = false
compiled_specializable(::Any) = true
compiled_specializable(accessors::Tuple) = all(compiled_specializable, accessors)
compiled_specializable(accessor::CompiledManyAccess) = compiled_specializable(accessor.inner)

@inline compiled_access_valid(values, slot::Int32, access::CompiledPackedAccess) = values.tags[slot] == access.pool
@inline compiled_access_valid(values, slot::Int32, ::CompiledBoxedAccess{T}) where {T} = iszero(values.tags[slot]) && values.boxed[slot] isa T
@inline compiled_access_valid(values, ::Nothing, ::Nothing) = true
@inline compiled_access_valid(values, slots::Tuple, access::Tuple) = all(map((slot, a) -> compiled_access_valid(values, slot, a), slots, access))
@inline compiled_access_valid(values, slots::CompiledManyOf, access::CompiledManyAccess) = compiled_access_valid(values, slots.slots, access.inner)

@inline function compiled_access_read(values, slot::Int32, access::CompiledPackedAccess{P}) where {P}
    pool = values.pools[access.pool]::P
    return compiled_unpack(pool, values.offsets[slot])
end
@inline compiled_access_read(values, slot::Int32, ::CompiledBoxedAccess{T}) where {T} = values.boxed[slot]::T
@inline compiled_access_read(values, ::Nothing, ::Nothing) = nothing
@inline compiled_access_read(values, slots::Tuple, access::Tuple) = map((slot, a) -> compiled_access_read(values, slot, a), slots, access)
@inline compiled_access_read(values, slots::CompiledManyOf, access::CompiledManyAccess) = ManyOf(compiled_access_read(values, slots.slots, access.inner))

@inline function compiled_access_write!(values, slot, access::CompiledPackedAccess{P}, result) where {P}
    if values.tags[slot] == access.pool && compiled_pack!(values.pools[access.pool]::P, values.offsets[slot], result)
        return nothing
    end
    values[slot] = result
    return nothing
end
@inline function compiled_access_write!(values, slot, ::CompiledBoxedAccess, result)
    values[slot] = result
    return nothing
end

struct CompiledTypedExecutor{A, O}
    access::A
    output::O
end
struct CompiledTypedPlan
    executors::Vector{Any}
    operation_executor::Vector{Int32}
end

@noinline function compiled_typed_evaluate!(executor, kernel, values, bindings, output)
    compiled_access_valid(values, bindings, executor.access) || return false
    inputs = compiled_access_read(values, bindings, executor.access)
    result = kernel(inputs)
    compiled_access_write!(values, output, executor.output, result)
    return true
end

function execute_compiled_typed!(plan::CompiledTypedPlan, program, index)
    id = plan.operation_executor[index]
    iszero(id) && return false
    operation = program.operations[index]
    return compiled_typed_evaluate!(plan.executors[id], program.kernels[operation.kernel],
        program.values, operation.inputs, operation.output)
end

function specialize_compiled_program!(program::CompiledProgram)
    program.values isa Vector{Any} || return program
    program.values = pack_compiled_values(program.values)
    program.values isa CompiledValues || return program
    executors = Any[]
    ids = Dict{Any, Int32}()
    operation_executor = zeros(Int32, length(program.operations))
    for (i, operation) in enumerate(program.operations)
        access = compiled_accessor(program.values, operation.inputs)
        compiled_specializable(access) || continue
        output = compiled_accessor(program.values, operation.output)
        executor = CompiledTypedExecutor(access, output)
        operation_executor[i] = get!(ids, executor) do
            push!(executors, executor)
            Int32(length(executors))
        end
    end
    program.typed = CompiledTypedPlan(executors, operation_executor)
    return program
end
