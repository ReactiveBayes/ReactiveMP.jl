@testitem "INVENTORY.md accounts for every node, export and engine hook" tags = [
    :quality
] begin
    # Phase P / open item #14: the package split is only safe if nothing falls through the
    # gap between "we meant to move it" and "we remembered to move it". `inventory.jl
    # --check` fails when an entity is missing from INVENTORY.md, still `undecided`, given
    # an invalid destination, stale, or deleted-while-exported with no migration note.
    #
    # Run in a subprocess against the package environment rather than the test
    # environment, because the script enumerates ReactiveMP's own surface and must not see
    # names the test target adds.
    root = dirname(@__DIR__)
    script = joinpath(root, "scripts", "inventory.jl")

    @test isfile(script)
    @test isfile(joinpath(root, "INVENTORY.md"))

    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(root) $(script) --check`
    process = run(ignorestatus(cmd))

    @test process.exitcode == 0
end
