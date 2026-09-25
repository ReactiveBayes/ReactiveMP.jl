using ReactiveMP, BayesBase, ExponentialFamily, Profile, InteractiveUtils
import ReactiveMP: MessageProductContext, compute_product_of_messages
v = randomvar(); ctx = MessageProductContext()
msgs = AbstractMessage[Message(NormalMeanVariance(randn(), 1.0 + rand()), false, false) for _ in 1:10]
f(n) = for _ in 1:n
    compute_product_of_messages(v, ctx, msgs)
end
f(10)
@time f(100000)
Profile.clear(); @profile f(300000)
Profile.print(IOContext(stdout, :displaysize => (200, 250)); format = :flat, sortedby = :count, mincount = 100000000)
