using RxInfer
# boxed captures in RxInfer methods: look for Core.Box in lowered code
function boxes(mod)
    out = String[]
    for name in names(mod; all = true)
        isdefined(mod, name) || continue
        f = getfield(mod, name)
        f isa Function || continue
        for m in methods(f)
            m.module === mod || continue
            try
                for ci in code_lowered(m.sig isa DataType ? f : f)
                end
            catch
            end
        end
    end
    return out
end
for m in methods(RxInfer.batch_inference) , m2 in (m,)
end
# kwbody methods are named like #batch_inference#NNN
for n in names(RxInfer; all = true)
    s = String(n)
    (occursin("batch_inference", s) || occursin("streaming_inference", s) || s == "infer" || occursin("#infer#", s)) || continue
    f = getfield(RxInfer, n)
    for m in methods(f)
        m.module === RxInfer || continue
        src = Base.uncompressed_ast(m)
        nbox = count(x -> x isa Expr && occursin("Core.Box", string(x)), src.code)
        boxed = Set{String}()
        for x in src.code
            str = string(x)
            if occursin("Core.Box", str)
                push!(boxed, first(str, 120))
            end
        end
        println(s, "  line ", m.line, "  Core.Box statements: ", nbox)
        for b in boxed
            println("    ", b)
        end
    end
end
println("--- boxed slot names")
for n in names(RxInfer; all = true)
    s = String(n)
    (startswith(s, "#batch_inference#") || startswith(s, "#streaming_inference#") || startswith(s, "#infer#")) || continue
    for m in methods(getfield(RxInfer, n))
        src = Base.uncompressed_ast(m)
        for x in src.code
            if x isa Expr && x.head === :(=) && x.args[1] isa Core.SlotNumber && occursin("Core.Box", string(x.args[2]))
                println(s, ": ", src.slotnames[x.args[1].id])
            end
        end
    end
end
