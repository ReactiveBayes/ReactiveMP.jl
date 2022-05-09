import JuliaFormatter;
using ArgParse;

s = ArgParseSettings()

@add_arg_table s begin
    "--overwrite"
    help = "overwrite your files with JuliaFormatter"
    action = :store_true
end

commandline_args = parse_args(s)
folders_to_format = ["scripts", "src"]

formatted = all(
    map(
        folder -> JuliaFormatter.format(folder, overwrite = commandline_args["overwrite"], verbose = true),
        folders_to_format
    )
)

if !formatted
    exit(1)
end
