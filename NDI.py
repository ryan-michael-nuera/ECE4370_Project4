# Class wrapping NDI tracker dll functions
# ECE 5370: Engineering for Surgery
# Fall 2024
# Author: Prof.Jack Noble
# jack.noble @ vanderbilt.edu
from ctypes import*
import numpy as np

class ndi:
    def __init__(self, ConfigFile):
        self.mydll = cdll.LoadLibrary('C:\\VISE5370_EngineeringForSurgery\\ndi.dll')
        self.mydll.Init.restype = c_void_p
        self.h = c_void_p(self.mydll.Init(create_string_buffer(ConfigFile)))

    def Start(self):
        self.mydll.Start(self.h)

    def Stop(self):
        self.mydll.Stop(self.h)

    def GetNumTools(self):
        n = c_int(self.mydll.GetNumTools(self.h))
        return n.value

    def GetTools(self, n):
        c_double()
        ToolMats = np.zeros([4, 4, n])
        ToolMats_c = np.zeros([n, 4, 4], dtype=c_double).ctypes.data_as(POINTER(c_double))
        self.mydll.GetTools(self.h, n, ToolMats_c)
        ToolMatsc = np.ctypeslib.as_array(ToolMats_c, shape=[n, 4, 4])
        for i in range(n):
            ToolMats[:, :, i] = ToolMatsc[i, :, :]
        return ToolMats

    def __del__(self):
        self.mydll.Delete(self.h)
        self.h = 0