
@testitem "getpoints: the multivariate generators reuse one buffer, by design" begin
    using ReactiveMP, BayesBase, LinearAlgebra

    import ReactiveMP:
        getpoints,
        getweights,
        GaussHermiteCubature,
        SphericalRadialCubature,
        srcubature,
        approximate_meancov

    # These tests pin an intentional, load-bearing performance contract rather than asserting
    # desired behaviour: the multivariate `getpoints` generators write every point into a single
    # preallocated vector and yield that same object each iteration (issue #633).
    #
    # It is not merely reuse. `approximate_meancov` *mutates* the yielded point in place --
    # `broadcast!(*, point, point, cv)` at approximations.jl -- to avoid allocating, which is
    # only safe because the next iteration rewrites the buffer before use. So the buffer is
    # shared in both directions, and changing `getpoints` to copy would silently make the
    # consumer's in-place trick pointless while adding an allocation per sigma point on a hot
    # path (GCV, delta nodes).
    #
    # The footgun is real, though: `collect` produces silently wrong cubature. These tests
    # document exactly that, so it is discoverable and so nobody "fixes" the reuse without
    # noticing the consumer depends on it.

    mean_vector = [1.0, 2.0]
    covariance  = [1.0 0.2; 0.2 2.0]

    @testset "collect() yields repeats of the last point -- do not do this" begin
        for method in (GaussHermiteCubature(3), srcubature())
            collected = collect(getpoints(method, mean_vector, covariance))

            @test length(collected) > 1
            # Every element is the identical object ...
            @test all(p -> p === collected[end], collected)
            # ... so only one distinct value survives.
            @test length(unique(collected)) == 1
        end
    end

    @testset "map(copy, ...) is the correct way to materialize" begin
        for method in (GaussHermiteCubature(3), srcubature())
            copied = map(copy, getpoints(method, mean_vector, covariance))

            @test length(unique(copied)) == length(copied)
            # No two points alias each other.
            for i in eachindex(copied), j in eachindex(copied)
                i == j || @test copied[i] !== copied[j]
            end
        end
    end

    @testset "lazy iteration -- how every in-repo consumer uses it -- is correct" begin
        # The weighted points must reproduce the distribution's own moments: with `g ≡ 1` the
        # cubature rule is exact for the mean and covariance of a Gaussian. This is what makes
        # the buffer reuse acceptable: consumed lazily, the points are right.
        for method in (GaussHermiteCubature(21), srcubature())
            weights = getweights(method, mean_vector, covariance)
            points  = getpoints(method, mean_vector, covariance)

            m̂ = zeros(2)
            for (w, p) in zip(weights, points)
                m̂ .+= w .* p
            end
            @test m̂ ≈ mean_vector atol = 1e-8

            # Recompute (the generator is single-pass over a shared buffer) for the covariance.
            weights = getweights(method, mean_vector, covariance)
            points  = getpoints(method, mean_vector, covariance)
            P̂       = zeros(2, 2)
            for (w, p) in zip(weights, points)
                d = p - mean_vector
                P̂ .+= w .* (d * d')
            end
            @test P̂ ≈ covariance atol = 1e-8
        end
    end

    @testset "the univariate Gauss-Hermite generator does not share a buffer" begin
        # It yields freshly computed scalars, so `collect` is safe there.
        collected = collect(getpoints(GaussHermiteCubature(3), 1.0, 2.0))
        @test length(unique(collected)) == length(collected)
    end

    @testset "approximate_meancov is exact for a linear g, buffer reuse notwithstanding" begin
        # End-to-end confirmation that the in-place consumer path is correct: for `g(x) = x`
        # the cubature-weighted mean is the distribution's mean exactly.
        for method in (GaussHermiteCubature(21), srcubature())
            m, P = approximate_meancov(
                method, (x) -> 1.0, mean_vector, covariance
            )
            @test m ≈ mean_vector atol = 1e-8
            @test P ≈ covariance atol = 1e-8
        end
    end
end
