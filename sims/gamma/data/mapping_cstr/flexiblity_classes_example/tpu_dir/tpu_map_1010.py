import numpy as np
mapping_cstr = {}
mapping_cstr["L2"] = {
                    "sp": np.array(["K"]),
                      # "order": ["K", "C", "Y","R", "S"],
                      "K": 16,
                      "Y": 16,
                      "C": 1,
                        "X":1,
                        "R":1,
                        "S":1,
                      }
mapping_cstr["L1"] = {"sp":np.array(["C"]),
                        # "sp_sz":16,
                      "order": ["K", "C", "Y", "R", "S", "X"],
                        "K":1,
                        "Y":1,
                        "C":1,
                        "X":1,
                        "R":1,
                        "S":1,
                    }

