import numpy as np
mapping_cstr = {}
mapping_cstr["L2"] = {"sp": np.array(["K"]),

                      }
mapping_cstr["L1"] = {"sp":np.array(["C"]),
                      "sp_sz":16,
                      "order": ["C", "Y", "X", "R", 'S', 'K'],
                      "K":"K",
                      "C":1,
                      "Y":"R",
                      "X":"S",
                      }