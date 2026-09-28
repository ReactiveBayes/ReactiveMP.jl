using RxInfer, Profile
@model function m1(y)
    θ ~ Beta(1.0, 1.0)
    y .~ Bernoulli(θ)
end
d = (y = [1.0, 0.0, 1.0],)
f0() = infer(model = m1(), data = d, session = nothing)
f0(); f0()
Profile.init(n = 10^7, delay = 0.00005)
Profile.clear()
@profile for _ in 1:20000
    f0()
end
Profile.print(IOContext(stdout, :displaysize => (10000, 250)); format = :flat, sortedby = :count, mincount = 400, C = false)
