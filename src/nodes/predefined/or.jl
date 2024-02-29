export OR

"""
OR node implements logic OR function (disjunction) that can be desribed by the followsing table:
| in1  in2 | out |
|  0    0  |  0  |
|  0    1  |  1  |
|  1    0  |  1  |
|  1    1  |  1  |
"""
struct OR end

@node OR Deterministic [out, in1, in2]
