#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

class PowerSimulator():
    def __init__(self, design):
        return None

    def power_model(self):
        print("comming soon")
        return 0

    def step(self):
        return 0

    def tick(self, clock_time):
        return 0

    def simulate(self, clock_time):
        self.tick(clock_time)
        self.step()
