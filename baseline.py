import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openvsp as vsp
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read

BASELINE_DIR = "./Baseline_Aircraft/"
DATABSE_PATH = "./Baseline_Aircraft/Database.xlsx"

class Baseline:
    """Class to handle baseline (kerosene) aircraft geometry
    and parameters. dispName must be one of:\n
    "Airbus A320-200ceo"\n
    (NO OTHER BASELINE AIRCRAFT IMPLEMENTED)
    """
    def __init__(self, dispName):
        self.dispName = dispName

        # Locate entry
        df = pd.read_excel(DATABSE_PATH)
        row = df.loc[df["Aircraft display name"] == dispName]
        
        # Extract aircraft parameters
        vsp3File = row["VSP3 file name"][0]
        self.dryCGxLoc = row["OE CG, m"][0]

        # OpenVSP API Reset
        vsp.VSPRenew()
        vsp.ClearVSPModel()
        vsp.DeleteAllResults()

        # Geometry import
        self.vspPath = BASELINE_DIR+dispName+"/"+vsp3File
        vsp.ReadVSPFile(self.vspPath)
        vsp.Update()

        # Establish OpenVSP objects
        self.vspVehicle = vsp.FindContainer("Vehicle", 0)
        self.fuse = vsp.FindGeom("Fuselage", 0)
        self.fuseXsecSurf = vsp.GetXSecSurf(self.fuse, 0)
        self.fuseXsecNum = vsp.GetNumXSec(self.fuseXsecSurf)

        self.fuseMaxH = 0
        self.fuseMaxD = 0
        for i in range(self.fuseXsecNum):
            xsec = vsp.GetXSec(self.fuseXsecSurf, i)
            self.fuseMaxH = max(self.fuseMaxH, vsp.GetXSecHeight(xsec))
            self.fuseMaxD = max(self.fuseMaxD, vsp.GetXSecWidth(xsec))

        ## SUAVE vehicle
        self.suaveVehicle = vsp_read(tag=self.vspPath,
                                     units_type="SI",
                                     specified_network=None,
                                     use_scaling=True)

        self.suaveVehicle["wings"]["main_wing"] = self.suaveVehicle["wings"]["wing"]
        del self.suaveVehicle["wings"]["wing"]
        self.suaveVehicle["wings"]["main_wing"].tag = "main_wing"

        self.suaveVehicle.mass_properties.max_takeoff = row["MTOW, kg"][0] * Units.kilogram
        self.suaveVehicle.mass_properties.max_zero_fuel = row["MZFW, kg"][0] * Units.kilogram

        #self.suaveVehicle.envelope.ultimate_load = 3.75 # UPDATE THESE - TAKEN FROM 737
        #self.suaveVehicle.envelope.limit_load = 2.5

        self.suaveVehicle.reference_area = self.suaveVehicle["wings"]["main_wing"].areas.reference * Units["meters**2"]

        #print(self.suaveVehicle.keys())
        self.suaveVehicle["networks"].clear()

        # Dry CG position?
        # Neutral trim centre of pressure position?
        # 
            
    def showVehicleGeom(self):
        SUAVE.Plots.Geometry.plot_vehicle(self.suaveVehicle, plot_control_points=False)
        plt.show()

    def report(self):
        print(f"\n### BASELINE TYPE {self.dispName} ###")
        print("## GEOMETRY SUMMARY ##")
        # Fuselage length and max height / dia
        # Wingspan, wing AR, wing area
        # Horz and vert stabiliser areas
        print("## WEIGHT AND BALANCE ##")
        # OE weight and OE CG position
        # Typical CG limits?
        print("## FUEL ##")
        # "mean" fuel moment arm, total capacity
        print("## ENGINE PERFORMANCE ##")
        # Mfr / model
        # Overall efficiency at cruise?
        # Max thrust at static / cruise conditions
        print("## FLIGHT OPERATING LIMITS ##")
        # Max Mach
        # Max dynamic pressure
        # Max level
        # Rotation and approach speeds
        print("## STRUCTURAL LIMITS ##")
        # Max landing weight
        # Max root bending moment
        # Fuselage shear load?
        print("## AERODYNAMIC ANALYSIS ##")
        # SUAVE 0 alpha, 0 beta, neutral trim incompressible L/D