module ReactiveMPTestingHelpers

export enabled_tests, addtests, key_to_filename, filename_to_key

enabled_tests = lowercase.(ARGS)

function addtests(filename)
    key = filename_to_key(filename)
    if isempty(enabled_tests) || key in enabled_tests
        include(filename)
    end
end

function key_to_filename(key)
    splitted = split(key, ":")
    return length(splitted) === 1 ? string("test_", first(splitted), ".jl") : string(join(splitted[1:end - 1], "/"), "/test_", splitted[end], ".jl")
end

function filename_to_key(filename)
    splitted   = split(filename, "/")
    if length(splitted) === 1
        return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
    else
        path, name = splitted[1:end - 1], splitted[end]
        return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
    end
end

end