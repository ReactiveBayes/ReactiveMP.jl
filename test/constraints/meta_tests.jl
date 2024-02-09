@testitem "Meta specification" begin
    using ReactiveMP, BayesBase, ExponentialFamily

    import ReactiveMP: MetaSpecification, MetaSpecificationOptions, MetaSpecificationEntry
    import ReactiveMP: resolve_meta

    @testset "resolve_meta #1" begin
        spec = MetaSpecification(
            (
                MetaSpecificationEntry(:NormalMeanPrecision, (:x, :z), 1),
                MetaSpecificationEntry(:NormalMeanPrecision, (:y, :z), 2),
                MetaSpecificationEntry(:NormalMeanVariance, (), 3)
            ),
            MetaSpecificationOptions(false)
        )

        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:a), randomvar(:b))) === nothing
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:b), randomvar(:x))) === nothing
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:b), randomvar(:z))) === nothing

        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:z))) === 1
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:y), randomvar(:z))) === 2

        @test resolve_meta(spec, NormalMeanVariance, (randomvar(:x), randomvar(:z))) === 3
        @test resolve_meta(spec, NormalMeanVariance, (randomvar(:y), randomvar(:z))) === 3
    end

    @testset "resolve_meta #2" begin
        spec = MetaSpecification((MetaSpecificationEntry(:NormalMeanPrecision, (:x, :z), 1), MetaSpecificationEntry(:NormalMeanPrecision, (), 2)), MetaSpecificationOptions(false))

        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:z))) === 1
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:y), randomvar(:z))) === 2

        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x),)) === 2
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:z),)) === 2
    end

    @testset "resolve_meta #3" begin
        spec = MetaSpecification(
            (MetaSpecificationEntry(:NormalMeanPrecision, (:x,), 1), MetaSpecificationEntry(:NormalMeanPrecision, (:x, :z), 2)), MetaSpecificationOptions(false)
        )

        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:z),)) === nothing
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x),)) === 1
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:z))) === 2
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:y), randomvar(:z))) === 2
    end

    @testset "resolve_meta #4" begin
        spec = MetaSpecification((MetaSpecificationEntry(:NormalMeanPrecision, (:x, :z), 1), MetaSpecificationEntry(:NormalMeanPrecision, (), 2)), MetaSpecificationOptions(false))

        @test resolve_meta(spec, NormalMeanPrecision, ((randomvar(:x), randomvar(:x)), randomvar(:z))) === 1
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), (randomvar(:z), randomvar(:z)))) === 1
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:y), randomvar(:z))) === 2
        @test resolve_meta(spec, NormalMeanPrecision, ((randomvar(:y), randomvar(:y)), randomvar(:z))) === 2
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:y), (randomvar(:z), randomvar(:z)))) === 2

        @test resolve_meta(spec, NormalMeanPrecision, ((randomvar(:x), randomvar(:x)),)) === 2
        @test resolve_meta(spec, NormalMeanPrecision, ((randomvar(:z), randomvar(:z)),)) === 2
    end

    @testset "resolve_meta ambiguity error #1" begin
        spec = MetaSpecification(
            (MetaSpecificationEntry(:NormalMeanPrecision, (:x, :z), 1), MetaSpecificationEntry(:NormalMeanPrecision, (:x, :y), 2)), MetaSpecificationOptions(false)
        )

        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:z))) === 1
        @test resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:y))) === 2
        @test_throws ErrorException resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:y), randomvar(:z)))
    end

    @testset "resolve_meta ambiguity error #2" begin
        spec = MetaSpecification(
            (MetaSpecificationEntry(:NormalMeanPrecision, (:x, :z), 1), MetaSpecificationEntry(:NormalMeanPrecision, (:z, :x), 2)), MetaSpecificationOptions(false)
        )

        @test_throws ErrorException resolve_meta(spec, NormalMeanPrecision, (randomvar(:x), randomvar(:z)))
        @test_throws ErrorException resolve_meta(spec, NormalMeanPrecision, (randomvar(:z), randomvar(:x)))
    end
end
