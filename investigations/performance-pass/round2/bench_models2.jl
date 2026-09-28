# Round 2's end-to-end benchmark: round 1's models and settings (`../bench_models.jl`, included),
# measured so that load on a shared machine and the placement of GC pauses do not decide the result.
#   julia --project=<env> bench_models2.jl <variant> <model> <tag> <outfile> [<posteriors-dir>]
#
# One TSV line per sample (not per summary), so that any statistic can be computed afterwards:
# (`ITER_SCALE`, default 2, multiplies round 1's I.)
#   variant tag model iters sample wall cpu gctime bytes pauses fullsweeps calib load1
# - `iters` is I or 2I for iterative models, alternating sample by sample so that drift affects
#   both equally; 0 for the time to first inference, and -1 for the load time.
# - `wall` is `time_ns`; `cpu` the thread's CPU time (`CLOCK_THREAD_CPUTIME_ID`), which leaves
#   out the time the process is descheduled while the machine is in use; `gctime` from `@timed`.
# - `pauses`, `fullsweeps`: GC collections during the sample (`Base.gc_num()` deltas).
# - `calib`: a fixed 20 ms loop timed just before the sample, the machine's current speed; `load1`
#   the one-minute load average. A sample whose `calib` is more than 10% above the process's best
#   is retried, up to three times, and kept with its record either way.
# No callbacks are set: v6 draws a `uuid4()` span id for every rule call whenever any callback is.

const NO_MAIN = true
include(joinpath(@__DIR__, "..", "bench_models.jl"))
import Distributions

# Round 2 runs twice round 1's iteration counts by default (20/40 for iid, gmm, hmm, linreg).
const ITER_SCALE = parse(Int, get(ENV, "ITER_SCALE", "2"))

const CLOCK_THREAD_CPUTIME_ID = Sys.isapple() ? Cint(16) : Cint(3)
function thread_cputime_ns()
    ts = Ref{NTuple{2, Clong}}((0, 0))
    ccall(:clock_gettime, Cint, (Cint, Ref{NTuple{2, Clong}}), CLOCK_THREAD_CPUTIME_ID, ts)
    return UInt64(ts[][1]) * 1_000_000_000 + UInt64(ts[][2])
end

# A fixed amount of scalar work, about 20 ms: its time is the machine's current speed.
@noinline function calibration_kernel(n)
    s = 0.0
    for i in 1:n
        s += sin(i * 1.0e-3)
    end
    return s
end
const CALIB_N = 2_000_000
calibrate() = (t = time_ns(); calibration_kernel(CALIB_N); (time_ns() - t) / 1.0e9)
const BEST_CALIB = Ref(Inf)

function one_sample(f)
    GC.gc()
    calib = calibrate()
    for _ in 1:3
        calib <= 1.1 * BEST_CALIB[] && break
        calib = calibrate()
    end
    BEST_CALIB[] = min(BEST_CALIB[], calib)
    load1 = Sys.loadavg()[1]
    g0 = Base.gc_num()
    c0 = thread_cputime_ns()
    s = @timed f()
    c1 = thread_cputime_ns()
    d = Base.GC_Diff(Base.gc_num(), g0)
    return (s.value, s.time, (c1 - c0) / 1.0e9, s.gctime, s.bytes, d.pause, d.full_sweep, calib, load1)
end

# Every parameter of every posterior, the type in full: the gate compares distributions, not
# summaries.
fullsummary(d) = d isa AbstractVector ? map(fullsummary, d) : fullsummary1(d)
function fullsummary1(d)
    p = try
        Distributions.params(d)
    catch
        try
            (mean(d), cov(d))
        catch
            string(d)
        end
    end
    return (string(typeof(d)), p)
end

function main2()
    run, iters1 = setup()
    iters = iters1 * ITER_SCALE
    for _ in 1:2
        calibrate()
    end
    BEST_CALIB[] = calibrate()
    rows = String[]
    row(iters, k, r) = push!(rows, join((VARIANT, TAG, MODELARG, iters, k, r[2:end]...), '\t'))
    push!(rows, join((VARIANT, TAG, MODELARG, -1, 0, TLOAD, 0, 0, 0, 0, 0, 0, 0), '\t'))
    it1 = max(iters, 1)
    first = one_sample(() -> run(it1))
    row(0, 0, first)
    result = first[1]
    if !isempty(PDIR)
        post = Dict(k => fullsummary(v) for (k, v) in pairs(posteriors_of(result)))
        serialize(joinpath(PDIR, "$(VARIANT)_$(MODELARG)_$(TAG).jls"), (post, free_energy_of(result)))
    end
    nsamples = parse(Int, get(ENV, "BENCH_SAMPLES", "11"))
    for k in 1:nsamples
        row(it1, k, one_sample(() -> run(it1)))
        iters > 0 && row(2iters, k, one_sample(() -> run(2iters)))
    end
    foreach(println, rows)
    return open(io -> foreach(l -> println(io, l), rows), OUT, "a")
end
main2()
