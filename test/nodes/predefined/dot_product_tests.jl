@testitem "dot node: default_meta" begin
    using ReactiveMP
    using MatrixCorrectionTools
    import ReactiveMP: default_meta, as_node_symbol

    # Check that `default_meta` for `dot` returns the correct type
    meta = default_meta(dot)
    @test meta isa MatrixCorrectionTools.ReplaceZeroDiagonalEntries

    # Check that it uses the expected `tiny` value
    @test meta.value == tiny
end
