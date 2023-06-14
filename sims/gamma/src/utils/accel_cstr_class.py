
class Accel_cstr():
    def __init__(self):
        self.accel_cstr = {}

    def set_cstr(self, level="L1", inbuffer="ScratchPad", outbuffer="ScratchPad", weightbuffer="ScratchPad", distrNoc="Bus", reduceNoc="Bus"):
        self.accel_cstr[level] = {
            "inbuf":inbuffer,
            "outbuf":outbuffer,
            "wbuf": weightbuffer,
            "distrNoc": distrNoc,
            "reduceNoc":reduceNoc
        }