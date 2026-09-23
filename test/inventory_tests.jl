@testitem "INVENTORY.md accounts for every node, export and engine hook" tags = [
    :quality,
] begin
    # Phase P / open item #14: the package split is only safe if nothing falls through the
    # gap between "we meant to move it" and "we remembered to move it". `inventory.jl
    # --check` fails when an entity is missing from INVENTORY.md, still `undecided`, given
    # an invalid destination, stale, or deleted-while-exported with no migration note.
    #
    # Run in a subprocess, in the v6 comparison environment: the inventory records where
    # everything in ReactiveMP 6.5.0 goes, and only v6.5.0 still has all of it.
    root = dirname(@__DIR__)
    script = joinpath(root, "scripts", "inventory.jl")

    @test isfile(script)
    @test isfile(joinpath(root, "INVENTORY.md"))

    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(joinpath(root, "compat", "v6-comparison")) $(script) --check`
    process = run(ignorestatus(cmd))

    @test process.exitcode == 0
end
