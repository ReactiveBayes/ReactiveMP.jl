export IMPLY

"""
IMPY node implements implication function that can be desribed by the followsing table:
| in1  in2 | out |
|  0    0  |  1  |
|  0    1  |  1  |
|  1    0  |  0  |
|  1    1  |  1  |
"""
struct IMPLY end

@node IMPLY Deterministic [out, in1, in2]
