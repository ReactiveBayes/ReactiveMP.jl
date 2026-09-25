# Variant "+lazy events": with no callbacks, the engine does not build the callback events at all.
# Adds `@invoke_callback` to callbacks.jl and uses it at every call site in message.jl and random.jl.
import sys, re, os
root = sys.argv[1]
p = os.path.join(root, 'src/callbacks.jl'); s = open(p).read()
anchor = "function invoke_callback(callbacks::Nothing, event::Event)"
assert anchor in s
s = s.replace(anchor, '''# Builds the event only when there are callbacks to hand it to: an event whose fields are
# inferred abstractly is instantiated at run time, which costs even when nothing listens.
macro invoke_callback(callbacks, event)
    return esc(:(let cb = $callbacks
        cb === nothing ? nothing : invoke_callback(cb, $event)
    end))
end

''' + anchor, 1)
open(p, 'w').write(s)
for f in ('src/message.jl', 'src/variables/random.jl'):
    p = os.path.join(root, f); s = open(p).read()
    n = len(re.findall(r'(?<![\w.@])invoke_callback\(', s))
    s = re.sub(r'(?<![\w.@])invoke_callback\(', '@invoke_callback(', s)
    open(p, 'w').write(s); print(f, n, 'sites')
