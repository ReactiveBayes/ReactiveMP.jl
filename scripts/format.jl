import Pkg;
Pkg.add("JuliaFormatter");

import JuliaFormatter;

for format_folder in ["scripts", "src"]
    if !(JuliaFormatter.format(format_folder, overwrite = false, verbose = true))
        exit(1)
    end
end
