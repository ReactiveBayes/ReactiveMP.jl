# Streaming VMP with @autoupdates and several iterations per data point: RxInfer's share of a tick.
#   julia --project=<env> prof_streaming.jl <variant> <outdir>
using RxInfer, StableRNGs, Profile
@model function kf_vmp(y, m_prev, v_prev, a_τ, b_τ)
    x_prev ~ Normal(mean = m_prev, variance = v_prev)
    τ ~ Gamma(shape = a_τ, rate = b_τ)
    x ~ Normal(mean = x_prev, variance = 1.0)
    y ~ Normal(mean = x, precision = τ)
end
rng = StableRNG(1); n = 500
y = cumsum(randn(rng, n)) .+ 0.5 .* randn(rng, n)
au = @autoupdates begin
    m_prev, v_prev = mean_var(q(x))
    a_τ, b_τ = params(q(τ))
end
init = @initialization begin
    q(x) = NormalMeanVariance(0.0, 1.0e3)
    q(τ) = GammaShapeRate(1.0, 1.0)
end
run() = infer(
    model = kf_vmp(), datastream = from(y) |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d,)), autoupdates = au,
    constraints = MeanField(), initialization = init, iterations = 5, keephistory = n, historyvars = (x = KeepLast(), τ = KeepLast()),
    autostart = true, free_energy = Float64, session = nothing
)
run(); run()
t = minimum([(GC.gc(); @elapsed run()) for _ in 1:5])
println("streaming kf_vmp n=$n, 5 it/point: ", round(t * 1.0e3; digits = 2), " ms  (", round(t / n * 1.0e6; digits = 2), " µs per data point)")
Profile.init(n = 10^7, delay = 0.0002); Profile.clear()
@profile for _ in 1:20
    run()
end
function shares()
    data, lidict = Profile.retrieve()
    bymod = Dict{String, Int}(); total = 0
    let i = 1
        while i <= length(data)
            j = i
            while j <= length(data) && data[j] != 0
                j += 1
            end
            frames = data[i:(j - 1)]; i = j + 1
            isempty(frames) && continue
            leaf = "C/runtime"
            for ip in frames
                sfs = get(lidict, ip, Base.StackTraces.StackFrame[])
                found = false
                for sf in sfs
                    sf.from_c && continue
                    f = String(sf.file)
                    leaf = occursin("RxInfer", f) ? "RxInfer" : occursin("ReactiveMP.jl/lib", f) ? "rules" : occursin("ReactiveMP", f) ? "ReactiveMP" : occursin("Rocket", f) ? "Rocket" : occursin("GraphPPL", f) ? "GraphPPL" : "Base/other"
                    found = true; break
                end
                found && break
            end
            bymod[leaf] = get(bymod, leaf, 0) + 1; total += 1
        end
    end
    for (k, v) in sort(collect(bymod); by = last, rev = true)
        println(rpad(k, 14), lpad(round(100v / total; digits = 1), 6), "%")
    end
    return
end
shares()
