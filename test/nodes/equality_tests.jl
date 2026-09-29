@testitem "EqualityChain's caches" tags = [:nodes] begin
    import ReactiveMP: EqualityChain, MessageObservable, AbstractMessage, EqualityLeftOutbound, EqualityRightOutbound,
        iscached, setcache!, ChainInvalidationCallback, MessageProductContext

    chain = EqualityChain([MessageObservable(AbstractMessage) for _ in 1:5], nothing, randomvar(), MessageProductContext())

    # One `Bool` per node, not bit-packed: writes to neighbouring nodes must not race when the
    # chain's messages are materialised from several threads.
    @test chain.cacheleft isa Vector{Bool} && chain.cacheright isa Vector{Bool}

    left, right = EqualityLeftOutbound(), EqualityRightOutbound()
    @test !any(i -> iscached(left, chain, i) || iscached(right, chain, i), 1:5)
    setcache!(left, chain, 1:3)
    setcache!(right, chain, 5:-1:2)
    @test [iscached(left, chain, i) for i in 1:5] == [true, true, true, false, false]
    @test [iscached(right, chain, i) for i in 1:5] == [false, true, true, true, true]

    # A new message at node 2 invalidates the left caches up to it and the right ones from it.
    ChainInvalidationCallback(2, chain)(nothing)
    @test [iscached(left, chain, i) for i in 1:5] == [false, false, true, false, false]
    @test [iscached(right, chain, i) for i in 1:5] == [false, false, false, false, false]
end

@testitem "EqualityChain computes each partial product once" tags = [:nodes] begin
    import ReactiveMP: EqualityChain, MessageObservable, AbstractMessage, EqualityLeftOutbound, EqualityRightOutbound,
        materialize!, set_initial_message!, ChainInvalidationCallback, getdata, MessageProductContext

    inputs = [MessageObservable(AbstractMessage) for _ in 1:5]
    foreach(((i, input),) -> set_initial_message!(input, Float64(i)), enumerate(inputs))
    products = Ref(0)
    sum_of_data(variable, context, pair) = (products[] += 1; Message(sum(x -> ismissing(x) ? 0.0 : x, getdata.(pair)), false, false))
    chain = EqualityChain(inputs, nothing, randomvar(), MessageProductContext(; fold_strategy = sum_of_data))
    left, right = EqualityLeftOutbound(), EqualityRightOutbound()

    # From an empty cache, the partial product at node k takes one product per node it covers.
    @test getdata(materialize!(right, chain, 4)) == 1 + 2 + 3 + 4
    @test products[] == 4
    @test getdata(materialize!(left, chain, 2)) == 2 + 3 + 4 + 5
    @test products[] == 8

    # A new message at node 3 leaves the right products at 1 and 2 and the left ones at 4 and 5
    # valid: only the invalidated ones are computed again.
    ChainInvalidationCallback(3, chain)(nothing)
    products[] = 0
    @test getdata(materialize!(right, chain, 4)) == 1 + 2 + 3 + 4
    @test products[] == 2
    products[] = 0
    @test getdata(materialize!(left, chain, 2)) == 2 + 3 + 4 + 5
    @test products[] == 2
end
