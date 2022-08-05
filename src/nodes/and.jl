export AND

"""
AND node implements logic AND function (conjuction) that can be desribed by the followsing table:
| in1  in2 | out |
|  0    0  |  0  |
|  0    1  |  0  |
|  1    0  |  0  |
|  1    1  |  1  |
"""
struct AND end

@node AND Deterministic [out, in1, in2]
