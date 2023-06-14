import numpy as np
mapping_cstr = {}
mapping_cstr["L2"] = {
    "sp": np.array(["Y"]),
    # "order":["Y", "X","C", "K", "R"],
    "Y":1,
    "X":1,
    "C":1,
    "K":16,
    "R":"R",
    "S":"S",
                      }
mapping_cstr["L1"] = {"sp":np.array(["R"]),
                      "sp2":np.array(["Y"]),
                        # "sp_sz":"R",
"order":["Y", "R","S", "X", "K", "C"],
                        "K":"K",
                        "C":1,
                        "Y":1,
                        "X":"X",
                        "R":1,
                        "S":"S",
                      }


