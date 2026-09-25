export MulticoreRunner

"""
    MulticoreRunner(; workers = Threads.nthreads(), min_batch_size = 32, min_work_ns = 200_000)

Run independent message rules of a single model on Julia's default thread pool.
Use with RxInfer as `options = (runner = MulticoreRunner(),)` and start Julia
with `--threads=auto`. A fresh execution state is created for each inference.

Ready rules run in waves: capture inputs, compute, then publish in emission
order on the calling task. Message rules, variable marginal products and local
marginal rules can run in parallel. Reactive subscriptions, equality-chain
caches and result delivery remain on the calling task. Narrow waves run serially. This is a different
message schedule from depth-first reactive execution; intermediate beliefs and
convergence rates may differ. It does not split datasets or duplicate models.

Rules with callbacks, annotation processors, or mutable metadata execute
serially, protecting shared mutable approximation state. Custom parallel rules
must treat their input distributions as read-only and must not mutate global
state. There is no guaranteed speedup for dependency chains or cheap rules.

`min_work_ns` is the minimum estimated numerical work in a wave before using
workers. A small serial sample of each job type estimates its cost, excluding
its first call to avoid charging compilation as execution. Samples are real
updates, never duplicate rule calls. Set `min_work_ns = 0` to disable calibration.
"""
struct MulticoreRunner
    workers::Int
    min_batch_size::Int
    min_work_ns::Int

    function MulticoreRunner(;
        workers::Integer = Threads.nthreads(),
        min_batch_size::Integer = 32,
        min_work_ns::Integer = 200_000,
    )
        1 <= workers <= Threads.nthreads() || throw(
            ArgumentError(
                "workers must be between 1 and Threads.nthreads() = $(Threads.nthreads()); start Julia with --threads=N",
            ),
        )
        min_batch_size >= 1 ||
            throw(ArgumentError("min_batch_size must be positive"))
        min_work_ns >= 0 ||
            throw(ArgumentError("min_work_ns must be nonnegative"))
        return new(workers, min_batch_size, min_work_ns)
    end
end

abstract type AbstractMulticoreJob end

struct MulticoreTerminalJob{A, V} <: AbstractMulticoreJob
    actor::A
    kind::Symbol
    value::V
    exception::Nothing
end

mutable struct MulticoreMapJob{A, F, D, R} <: AbstractMulticoreJob
    actor::A
    mapping::F
    input::D
    parallel::Bool
    result::Union{Nothing, R}
    exception::Any
    done::Bool

    # R cannot be inferred from an initial `nothing` result. Require the
    # explicit result type already supplied by the stream's map operator.
    function MulticoreMapJob{A, F, D, R}(actor, mapping, input, parallel, result, exception, done) where {A, F, D, R}
        return new{A, F, D, R}(actor, mapping, input, parallel, result, exception, done)
    end
end

mutable struct MulticoreJob{A, F, M, Q} <: AbstractMulticoreJob
    actor::A
    mapping::F
    messages::M
    marginals::Q
    parallel::Bool
    result::Union{Nothing, Message}
    exception::Any
    done::Bool
end

mutable struct MulticoreStreamPostprocessor <: AbstractStreamPostprocessor
    config::MulticoreRunner
    pending::Vector{AbstractMulticoreJob}
    current::Vector{AbstractMulticoreJob}
    active::Bool
    draining::Bool
    waves::Int
    parallel_jobs::Int
    serial_jobs::Int
    costs::IdDict{DataType, Float64}
end

instantiate_runner(config::MulticoreRunner) = MulticoreStreamPostprocessor(
    config,
    AbstractMulticoreJob[],
    AbstractMulticoreJob[],
    false,
    false,
    0,
    0,
    0,
    IdDict{DataType, Float64}(),
)
instantiate_runner(::Nothing) = nothing
instantiate_runner(config) = throw(
    ArgumentError(
        "Unsupported runner $(typeof(config)); expected MulticoreRunner() or nothing",
    ),
)

# Lifecycle fallbacks keep existing postprocessors unchanged.
start_runner!(::Any) = nothing
stop_runner!(::Any) = nothing
synchronize_runner!(::Any) = nothing
start_runner!(p::CompositeStreamPostprocessor) =
    foreach(start_runner!, p.stages)
stop_runner!(p::CompositeStreamPostprocessor) = foreach(stop_runner!, p.stages)
synchronize_runner!(p::CompositeStreamPostprocessor) =
    foreach(synchronize_runner!, p.stages)
start_runner!(runner::MulticoreStreamPostprocessor) =
    (runner.active = true; nothing)
function stop_runner!(runner::MulticoreStreamPostprocessor)
    runner.active = false
    empty!(runner.pending)
    empty!(runner.current)
    return nothing
end

struct MulticoreMessageProxy <: Rocket.ActorProxy
    runner::MulticoreStreamPostprocessor
end

struct MulticoreMessageActor{L, A} <: Rocket.Actor{L}
    runner::MulticoreStreamPostprocessor
    actor::A
end

Rocket.actor_proxy!(
    ::Type{L}, proxy::MulticoreMessageProxy, actor::A
) where {L, A} = MulticoreMessageActor{L, A}(proxy.runner, actor)

function postprocess_stream_of_outbound_messages(
    runner::MulticoreStreamPostprocessor, stream
)
    T = Rocket.subscribable_extract_type(stream)
    return if T <: AbstractMessage
        Rocket.proxy(T, stream, MulticoreMessageProxy(runner))
    else
        stream
    end
end

postprocess_stream_of_marginals(::MulticoreStreamPostprocessor, stream) = stream
postprocess_stream_of_scores(::MulticoreStreamPostprocessor, stream) = stream

# Computation hooks also cover eager marginal/product mappings. The serial
# fallback uses the original Rocket operators, with no scheduling overhead.
runner_map(::Any, ::Type{R}, stream, mapping, parallel::Bool) where {R} =
    stream |> Rocket.MapOperator{R, typeof(mapping)}(mapping)
function runner_map(
    p::CompositeStreamPostprocessor, ::Type{R}, stream, mapping, parallel::Bool
) where {R}
    runner = find_multicore_runner(p)
    return runner_map(runner, R, stream, mapping, parallel)
end
find_multicore_runner(::Any) = nothing
find_multicore_runner(p::MulticoreStreamPostprocessor) = p
function find_multicore_runner(p::CompositeStreamPostprocessor)
    for stage in p.stages
        runner = find_multicore_runner(stage)
        isnothing(runner) || return runner
    end
    return nothing
end

struct MulticoreMapProxy{L, R, F} <: Rocket.ActorProxy
    runner::MulticoreStreamPostprocessor
    mapping::F
    parallel::Bool
end
struct MulticoreMapActor{L, R, A, F} <: Rocket.Actor{L}
    runner::MulticoreStreamPostprocessor
    actor::A
    mapping::F
    parallel::Bool
end
Rocket.actor_proxy!(
    ::Type, p::MulticoreMapProxy{L, R, F}, actor::A
) where {L, R, F, A} =
    MulticoreMapActor{L, R, A, F}(p.runner, actor, p.mapping, p.parallel)
function runner_map(
    runner::MulticoreStreamPostprocessor,
    ::Type{R},
    stream,
    mapping::F,
    parallel::Bool,
) where {R, F}
    L = Rocket.subscribable_extract_type(stream)
    return Rocket.proxy(
        R, stream, MulticoreMapProxy{L, R, F}(runner, mapping, parallel)
    )
end
function Rocket.on_next!(actor::MulticoreMapActor{L, R}, input) where {L, R}
    if !actor.runner.active
        return Rocket.next!(actor.actor, actor.mapping(input))
    end
    push!(
        actor.runner.pending,
        MulticoreMapJob{
            typeof(actor.actor), typeof(actor.mapping), typeof(input), R
        }(
            actor.actor,
            actor.mapping,
            input,
            actor.parallel,
            nothing,
            nothing,
            false,
        ),
    )
    return nothing
end
Rocket.on_error!(actor::MulticoreMapActor, err) =
    multicore_terminal!(actor.runner, actor.actor, :error, err)
Rocket.on_complete!(actor::MulticoreMapActor) =
    multicore_terminal!(actor.runner, actor.actor, :complete, nothing)

runner_collect_latest(
    ::Any, ::Type{T}, ::Type{R}, sources, mapping, callback, parallel
) where {T, R} = collectLatest(T, R, sources, mapping, callback)
function runner_collect_latest(
    p::Union{MulticoreStreamPostprocessor, CompositeStreamPostprocessor},
    ::Type{T},
    ::Type{R},
    sources,
    mapping,
    callback,
    parallel,
) where {T, R}
    runner = find_multicore_runner(p)
    isnothing(runner) && return runner_collect_latest(
        nothing, T, R, sources, mapping, callback, parallel
    )
    # collectLatest reuses its storage; copy before handing ownership to a job.
    inputs = collectLatest(
        T,
        Vector{Message},
        sources,
        snapshot_multicore_product,
        reset_multicore_product_status,
    )
    return runner_map(runner, R, inputs, mapping, parallel)
end
snapshot_multicore_product(messages) =
    Message[as_message(message) for message in messages]
function reset_multicore_product_status(wrapper, messages)
    if !all(is_clamped, messages) && all(is_clamped_or_initial, messages)
        Rocket.fill_vstatus!(wrapper, true)
    end
    return nothing
end

struct MulticoreSnapshot{T}
    value::T
end
Rocket.getrecent(snapshot::MulticoreSnapshot) = snapshot.value
snapshot_multicore_dependencies(sources) =
    map(source -> MulticoreSnapshot(getrecent(source)), sources)
function reset_multicore_marginal_status(wrapper, snapshots)
    inputs = map(getrecent, snapshots)
    if !all(x -> __check_all(is_clamped, x), inputs) &&
        all(x -> __check_all(is_clamped_or_initial, x), inputs)
        Rocket.fill_vstatus!(wrapper, true)
    end
    return nothing
end
runner_combine_latest_updates(
    ::Any, sources, strategy, ::Type{R}, mapping, callback
) where {R} = combineLatestUpdates(sources, strategy, R, mapping, callback)
function runner_combine_latest_updates(
    p::Union{MulticoreStreamPostprocessor, CompositeStreamPostprocessor},
    sources,
    strategy,
    ::Type{R},
    mapping::MarginalMapping,
    callback,
) where {R}
    runner = find_multicore_runner(p)
    isnothing(runner) && return runner_combine_latest_updates(
        nothing, sources, strategy, R, mapping, callback
    )
    inputs = combineLatestUpdates(
        sources,
        strategy,
        Any,
        snapshot_multicore_dependencies,
        reset_multicore_marginal_status,
    )
    parallel = multicore_readonly(mapping.meta) && isnothing(mapping.factornode)
    return runner_map(runner, R, inputs, mapping, parallel)
end

Rocket.on_next!(actor::MulticoreMessageActor, message) =
    Rocket.next!(actor.actor, message)
function Rocket.on_next!(actor::MulticoreMessageActor, message::DeferredMessage)
    runner = actor.runner
    if !runner.active
        return Rocket.next!(actor.actor, message)
    end
    # Snapshot on the publisher task: workers must never read mutable Rocket
    # recent-value caches while a different wave is being delivered.
    messages = Rocket.getrecent(message.messages)
    marginals = Rocket.getrecent(message.marginals)
    push!(
        runner.pending,
        MulticoreJob(
            actor.actor,
            message.mappingFn,
            messages,
            marginals,
            multicore_parallel_safe(message.mappingFn),
            nothing,
            nothing,
            false,
        ),
    )
    return nothing
end
Rocket.on_error!(actor::MulticoreMessageActor, err) =
    multicore_terminal!(actor.runner, actor.actor, :error, err)
Rocket.on_complete!(actor::MulticoreMessageActor) =
    multicore_terminal!(actor.runner, actor.actor, :complete, nothing)

function multicore_terminal!(runner, actor, kind, value)
    job = MulticoreTerminalJob(actor, kind, value, nothing)
    if runner.active
        push!(runner.pending, job)
    else
        publish_multicore_job!(job)
    end
    return nothing
end
compute_multicore_job!(::MulticoreTerminalJob) = nothing
multicore_parallel_safe(::MulticoreTerminalJob) = false
function publish_multicore_job!(job::MulticoreTerminalJob)
    return if job.kind === :complete
        Rocket.complete!(job.actor)
    else
        Rocket.error!(job.actor, job.value)
    end
end

# A conservative fallback is essential: e.g. importance-sampling metadata
# owns reusable buffers and RNGs. Unknown mappings also stay serial.
multicore_parallel_safe(::Any) = false
multicore_parallel_safe(mapping::MessageMapping) =
    multicore_readonly(mapping.meta) &&
    multicore_readonly(mapping.vconstraint) &&
    isnothing(mapping.annotations) &&
    isnothing(mapping.callbacks) &&
    isnothing(mapping.rulefallback) &&
    isnothing(mapping.factornode)

# Symbols and type objects are immutable but not isbits. Recursively inspect
# immutable wrappers so a tuple/closure containing an RNG or a buffer is still
# rejected. This checks captured state, not arbitrary side effects of user code.
function multicore_readonly(value)
    T = typeof(value)
    isbitstype(T) && return true
    value isa Union{Symbol, String, Type} && return true
    ismutabletype(T) && return false
    return all(
        i -> isdefined(value, i) && multicore_readonly(getfield(value, i)),
        1:fieldcount(T),
    )
end
multicore_parallel_safe(context::MessageProductContext) =
    isnothing(context.callbacks) &&
    isnothing(context.annotations) &&
    multicore_readonly(context)

multicore_parallel_safe(::Any, context) = false
multicore_parallel_safe(::MulticoreStreamPostprocessor, context) =
    multicore_parallel_safe(context)
multicore_parallel_safe(p::CompositeStreamPostprocessor, context) =
    multicore_parallel_safe(find_multicore_runner(p), context)

function compute_multicore_job!(job::MulticoreJob)
    try
        job.result = job.mapping(job.messages, job.marginals)
    catch exception
        job.exception = exception
    end
    job.done = true
    return nothing
end

function compute_multicore_job!(job::MulticoreMapJob)
    try
        job.result = job.mapping(job.input)
    catch exception
        job.exception = exception
    end
    job.done = true
    return nothing
end

publish_multicore_job!(job::MulticoreJob) =
    Rocket.next!(job.actor, job.result::Message)
multicore_parallel_safe(job::MulticoreJob) = job.parallel
multicore_parallel_safe(job::MulticoreMapJob) = job.parallel
publish_multicore_job!(job::MulticoreMapJob{A, F, D, R}) where {A, F, D, R} =
    Rocket.next!(job.actor, job.result::R)

multicore_job_done(job::Union{MulticoreJob, MulticoreMapJob}) = job.done
multicore_job_done(::MulticoreTerminalJob) = false

function estimate_multicore_work!(runner, jobs)
    iszero(runner.config.min_work_ns) && return Inf
    samples = IdDict{DataType, Vector{AbstractMulticoreJob}}()
    for job in jobs
        multicore_parallel_safe(job) || continue
        T = typeof(job)
        haskey(runner.costs, T) && continue
        sample = get!(() -> AbstractMulticoreJob[], samples, T)
        length(sample) < 9 && push!(sample, job)
    end
    for (T, sample) in samples
        # The first call may compile this specialization. Do not time it, and
        # never replay a sampled update (rules may own expensive output buffers).
        compute_multicore_job!(first(sample))
        if length(sample) > 1
            started = time_ns()
            for i in 2:length(sample)
                compute_multicore_job!(sample[i])
            end
            runner.costs[T] =
                Float64(time_ns() - started) / (length(sample) - 1)
        else
            runner.costs[T] = Inf
        end
        runner.serial_jobs += length(sample)
    end
    work = 0.0
    for job in jobs
        if multicore_parallel_safe(job) && !multicore_job_done(job)
            work += runner.costs[typeof(job)]
        end
    end
    return work
end

function synchronize_runner!(runner::MulticoreStreamPostprocessor)
    runner.draining && return nothing
    runner.draining = true
    try
        while !isempty(runner.pending)
            runner.current, runner.pending = runner.pending, runner.current
            jobs = runner.current
            n = length(jobs)
            runner.waves += 1
            eligible =
                if runner.config.workers > 1 &&
                    n >= runner.config.min_batch_size
                    count(multicore_parallel_safe, jobs)
                else
                    0
                end
            parallel =
                eligible >= runner.config.min_batch_size &&
                estimate_multicore_work!(runner, jobs) >=
                runner.config.min_work_ns
            if parallel
                # Coarse contiguous chunks amortize task creation. Unsafe jobs
                # run after joining all workers, before any result is published.
                runner.parallel_jobs += count(
                    job ->
                        multicore_parallel_safe(job) &&
                        !multicore_job_done(job),
                    jobs,
                )
                nchunks = min(runner.config.workers, eligible)
                @sync for chunk in 1:nchunks
                    lo = fld((chunk - 1) * n, nchunks) + 1
                    hi = fld(chunk * n, nchunks)
                    Threads.@spawn for i in lo:hi
                        job = jobs[i]
                        multicore_parallel_safe(job) &&
                            !multicore_job_done(job) &&
                            compute_multicore_job!(job)
                    end
                end
                for job in jobs
                    if !multicore_parallel_safe(job)
                        compute_multicore_job!(job)
                        runner.serial_jobs += 1
                    end
                end
            else
                for job in jobs
                    if !multicore_job_done(job)
                        compute_multicore_job!(job)
                        runner.serial_jobs += 1
                    end
                end
            end
            # A failed wave must not publish a partially computed result.
            for job in jobs
                isnothing(job.exception) || throw(job.exception)
            end
            foreach(publish_multicore_job!, jobs)
            empty!(jobs)
        end
    finally
        runner.draining = false
    end
    return nothing
end
