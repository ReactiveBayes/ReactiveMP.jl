using Serialization
dir = ARGS[1]
ok = true
for m in ("ssm1", "iid", "hmm", "nl", "gmm", "filter")
    a = deserialize(joinpath(dir, "P12_$(m)_gate.jls")); b = deserialize(joinpath(dir, "X1_$(m)_gate.jls"))
    same = isequal(a, b)
    global ok &= same
    println(rpad(m, 8), same ? "bit-identical" : "DIFFERENT")
end
exit(ok ? 0 : 1)
