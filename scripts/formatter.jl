# Formatting, via Runic.
#
# Runic is deterministic and has no configuration, so there is nothing to disagree about and
# no style file to drift. That is the whole reason it replaced JuliaFormatter, whose output
# moved between minor releases and between Julia versions -- CI and contributors could format
# the same code differently with no change in the code.
#
#   julia --project=scripts scripts/formatter.jl              # check only, non-zero on diff
#   julia --project=scripts scripts/formatter.jl --overwrite   # rewrite in place

using Runic
using ArgParse

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

# `docs/` is excluded, as it was under JuliaFormatter's `ignore = ["docs"]`. Documenter's own
# `@example` blocks are formatted by nobody and reformatting `make.jl` buys nothing.
const EXCLUDED_DIRS = ("docs", ".git")

function julia_files(root)
    files = String[]
    for (dir, subdirs, names) in walkdir(root)
        filter!(d -> !(d in EXCLUDED_DIRS), subdirs)
        for name in names
            endswith(name, ".jl") && push!(files, joinpath(dir, name))
        end
    end
    return sort(files)
end

function main()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--overwrite"
        help = "Rewrite the files in place instead of only reporting"
        action = :store_true
    end
    overwrite = parse_args(ARGS, settings)["overwrite"]

    unformatted = String[]
    failed = String[]

    for path in julia_files(PROJECT_ROOT)
        source = read(path, String)
        formatted = try
            Runic.format_string(source)
        catch err
            push!(failed, path)
            continue
        end
        formatted == source && continue
        push!(unformatted, relpath(path, PROJECT_ROOT))
        overwrite && write(path, formatted)
    end

    if !isempty(failed)
        @error "Runic could not parse some files; formatting is incomplete" files = failed
        exit(2)
    end

    if isempty(unformatted)
        @info "Codestyle checks have passed"
    elseif overwrite
        @info "Runic reformatted $(length(unformatted)) file(s)"
    else
        @error """
        Runic check has failed for $(length(unformatted)) file(s). Run `make format` from the
        main directory and commit the result.
        """ files = first(unformatted, 25)
        exit(1)
    end
    return nothing
end

main()
