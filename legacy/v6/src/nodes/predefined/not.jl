export NOT

"""
NOT node implements negation function that can be desribed by the followsing table:
| in  | out |
|  0  |  1  |
|  1  |  0  |
"""
struct NOT end

@node NOT Deterministic [out, in]
