export IMPLY

struct IMPLY end

#= IMPY node implements implication function that can be desribed by the followsing table:
| in1  in2 | out |
|  0    0  |  1  |
|  0    1  |  1  |
|  1    0  |  0  |
|  1    1  |  1  |
=#
@node IMPLY Deterministic [out, in1, in2]
