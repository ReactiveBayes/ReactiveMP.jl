@testitem "multiplication node: default_meta" begin
    using ReactiveMP
    using MatrixCorrectionTools
    import ReactiveMP: default_meta

    # Check that `default_meta` for `multiplication` returns the correct type
    meta = default_meta(*)
    @test meta isa MatrixCorrectionTools.ReplaceZeroDiagonalEntries

    # Check that it uses the expected `tiny` value
    @test meta.value == tiny
end
