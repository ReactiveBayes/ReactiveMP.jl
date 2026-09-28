# The longest chain that activates without a StackOverflowError, on a task with a fixed stack,
# for the engine's chain graph and for RxInfer's ssm1 without `limit_stack_depth`.
#   julia --project=<env> stack_depth.jl <variant> <outfile> [stack MiB = 8]
# Also reports the stack bytes per chain link (stack size / longest chain) and the activation time
# at a length every variant reaches.

using ReactiveMP, Rocket, ExponentialFamily, BayesBase, StandardMessagePassingRules, RxInfer, Statistics
import ReactiveMP: activate!, FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions, israndom, get_stream_of_marginals

const VARIANT = ARGS[1]
const OUT = ARGS[2]
const STACK = parse(Int, get(ARGS, 3, "8")) * 2^20

bethe(interfaces) = (Tuple(n for (n, v) in interfaces if israndom(v)), Tuple((n,) for (n, v) in interfaces if !israndom(v))...)
mknode(fform, interfaces) = factornode(fform, interfaces, filter(!isempty, bethe(interfaces)))

function chain_activate(n)
    x = [randomvar() for _ in 1:n]; y = [datavar() for _ in 1:n]; v = constvar(1.0)
    nodes = Any[mknode(NormalMeanVariance, [(:out, x[1]), (:μ, constvar(0.0)), (:v, constvar(10.0))])]
    push!(nodes, mknode(NormalMeanVariance, [(:out, y[1]), (:μ, x[1]), (:v, v)]))
    for i in 2:n
        push!(nodes, mknode(NormalMeanVariance, [(:out, x[i]), (:μ, x[i - 1]), (:v, v)]))
        push!(nodes, mknode(NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, v)]))
    end
    foreach(v -> activate!(v, RandomVariableActivationOptions()), x)
    foreach(v -> activate!(v, DataVariableActivationOptions()), y)
    foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
    subs = [subscribe!(get_stream_of_marginals(w), (_) -> nothing) for w in x]
    foreach(new_observation!, y, randn(n))
    foreach(unsubscribe!, subs)
    return nothing
end

@model function ssm1(y, P)
    x_prior ~ Normal(mean = 0.0, variance = 10000.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 1.0)
        y[i] ~ Normal(mean = x[i], variance = P)
        x_prev = x[i]
    end
end
ssm1_infer(n; options = NamedTuple()) = (infer(model = ssm1(P = 10.0), data = (y = randn(n),), options = options, session = nothing); nothing)

# runs f(n) on a task with a STACK-byte stack; true when it finished, false on StackOverflowError
function fits(f, n)
    t = Task(() -> Base.CoreLogging.with_logger(() -> f(n), Base.CoreLogging.NullLogger()), STACK)
    schedule(t)
    try
        wait(t)
        return true
    catch e
        err = e isa TaskFailedException ? e.task.exception : e
        (err isa StackOverflowError || occursin("StackOverflow", sprint(showerror, err))) && return false
        rethrow()
    end
end

function longest(f; lo = 16, hi = parse(Int, get(ENV, "STACK_HI", "32768")))
    fits(f, lo) || return 0
    # grow, then bisect
    n = lo
    while n < hi && fits(f, 2n)
        n *= 2
    end
    n >= hi && return hi
    a, b = n, 2n
    while b - a > max(8, a ÷ 50)
        m = (a + b) ÷ 2
        fits(f, m) ? (a = m) : (b = m)
    end
    return a
end

function timed(f, n; reps = 5)
    fits(f, n)
    ts = [(GC.gc(); @elapsed fits(f, n)) for _ in 1:reps]
    return minimum(ts), median(ts)
end

rows = String[]
function record(name, value, extra = "")
    line = join((VARIANT, name, value, extra), '\t'); println(line)
    return push!(rows, line)
end

nchain = longest(chain_activate)
record("engine chain: longest n (stack $(STACK >> 20) MiB)", nchain, round(STACK / max(nchain, 1); digits = 0))
nssm = longest(ssm1_infer)
record("RxInfer ssm1, no limit_stack_depth: longest n", nssm, round(STACK / max(nssm, 1); digits = 0))
for n in (200, 1000)
    t = timed(chain_activate, n); record("engine chain n=$n activate+1 pass (s)", t[1], t[2])
    t = timed(ssm1_infer, n); record("RxInfer ssm1 n=$n infer (s)", t[1], t[2])
    t = timed(m -> ssm1_infer(m; options = (limit_stack_depth = 100,)), n); record("RxInfer ssm1 n=$n infer limit_stack_depth=100 (s)", t[1], t[2])
end
open(io -> foreach(r -> println(io, r), rows), OUT, "a")
