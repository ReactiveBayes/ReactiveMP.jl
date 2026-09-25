@testitem "EqualityChain's caches" tags = [:nodes] begin
    import ReactiveMP: EqualityChain, MessageObservable, AbstractMessage, EqualityLeftOutbound, EqualityRightOutbound,
        iscached, setcache!, ChainInvalidationCallback

    chain = EqualityChain([MessageObservable(AbstractMessage) for _ in 1:5], nothing, identity)

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
