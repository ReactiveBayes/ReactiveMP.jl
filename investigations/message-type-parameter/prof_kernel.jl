using ReactiveMP, BayesBase, ExponentialFamily, BenchmarkTools
import ReactiveMP: MessageProductContext, product_kernel, AnnotationDict
v = randomvar(); ctx = MessageProductContext()
a = NormalMeanVariance(0.0, 1.0); b = NormalWeightedMeanPrecision(0.0, 1.0); ann = AnnotationDict()
@noinline call(v, ctx, x::Ref{Any}, y::Ref{Any}, ann) = product_kernel(v, ctx, x[], y[], ann, ann)
rx, ry = Ref{Any}(b), Ref{Any}(a)
println("kernel, known types:  ", @belapsed product_kernel($v, $ctx, $b, $a, $ann, $ann))
println("kernel, via Any:      ", @belapsed call($v, $ctx, $rx, $ry, $ann))
@noinline k2(v, ctx, x, y) = (x, y)
@noinline call2(v, ctx, x::Ref{Any}, y::Ref{Any}) = k2(v, ctx, x[], y[])
println("trivial 4-arg kernel via Any: ", @belapsed call2($v, $ctx, $rx, $ry))
@noinline k3(x, y) = (x, y)
@noinline call3(x::Ref{Any}, y::Ref{Any}) = k3(x[], y[])
println("trivial 2-arg kernel via Any: ", @belapsed call3($rx, $ry))
@noinline k4(v, x, y) = (x, y)
@noinline call4(v, x::Ref{Any}, y::Ref{Any}) = k4(v, x[], y[])
println("trivial 3-arg kernel (randomvar, Any, Any): ", @belapsed call4($v, $rx, $ry))
@noinline k5(c, x, y) = (x, y)
@noinline call5(c, x::Ref{Any}, y::Ref{Any}) = k5(c, x[], y[])
println("trivial 3-arg kernel (ctx, Any, Any): ", @belapsed call5($ctx, $rx, $ry))
println(typeof(ctx))
xs = Any[a, b, b, b, b, b, b, b, b]
@noinline function loop(v, ctx, xs, y, ann)
    for x in xs
        product_kernel(v, ctx, x, y, ann, ann)
    end
    return
end
println("kernel via Any, 9 calls, alternating first-arg types: ", @belapsed loop($v, $ctx, $xs, $a, $ann))
xs2 = Any[b for _ in 1:9]
println("kernel via Any, 9 calls, one signature: ", @belapsed loop($v, $ctx, $xs2, $a, $ann))
@noinline function loopy(v, ctx, xs, y::Ref{Any}, ann)
    for x in xs
        product_kernel(v, ctx, x, y[], ann, ann)
    end
    return
end
println("kernel via Any (both args Any), 9 calls, alternating: ", @belapsed loopy($v, $ctx, $xs, $ry, $ann))
