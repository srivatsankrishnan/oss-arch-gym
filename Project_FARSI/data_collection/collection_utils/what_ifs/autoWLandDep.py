#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

depths = [3, 5, 8, 10]
workloads = [{"edge_detection"}, {"hpvm_cava"}, {"audio_decoder"}, {"edge_detection", "hpvm_cava"}, {"edge_detection", "audio_decoder"}, {"hpvm_cava", "audio_decoder"}, {"audio_decoder", "edge_detection", "hpvm_cava"}]

for d in depths:
    for w in workloads:
        print(d)
        print(w)
        print("\n")
    print("\n")