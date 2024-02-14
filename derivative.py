import numpy as np
import pandas as pd
import pandas as pd

import baseline as base

class Derivative:
    def __init__(self, baseline, modName):
        """Class defining derivatives of Baseline aircraft, including OpenVSP geometry.

        Args:
            baseline (Baseline): Instance of Baseline class.
            modName (string): Derivative modification tag.
        """

        self.baseline = baseline
        self.modName = modName

        self.dispName = self.baseline.dispName + "_" + self.modName
    
    def updateDerivGeom(tankType="dorsal", tankCapacity=10)
        # Create new derived OpenVSP
        pass

   def updateSUAVEvehicle():
        # check if createDerivGeom has been run at least once

