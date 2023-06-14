import numpy as np
mapping_cstr = {}
mapping_cstr["L2"] = {"sp": np.array(["K"]),
                      "order": ["K", "C", "Y", "X", "R"],
                      "K": 1,
                      "C": 64,
                      "Y": "R",
                      "X": "S",
                      "R": "R",
                      "S": "S",
                      }
mapping_cstr["L1"] = {"sp":np.array(["C"]),
                        # "sp_sz":64,
                      "order": ["K","C", "Y", "X", "R", "S"],
                        "K":"K",
                        "C":1,
                        "Y":"R",
                        "X":"S",
                        "R":"R",
                        "S":"S",
                    }

