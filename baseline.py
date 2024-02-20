import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import openvsp as vsp
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read

BASELINE_DIR = "./Baseline_Aircraft/"
DATABASE_PATH = "./Baseline_Aircraft/Database.xlsx"

class Baseline:
    f"""Class to handle baseline (kerosene) aircraft geometry
    and parameters. dispName must be one of: \n
    "Airbus A320-200ceo"\n
    (NO OTHER BASELINE AIRCRAFT IMPLEMENTED)
    """

    def __init__(self, dispName):
        self.dispName = dispName
        self.fuels = {"Jet-A": SUAVE.Attributes.Propellants.Jet_A(),
                      "LH2": SUAVE.Attributes.Propellants.Liquid_H2()}
        self.dispNames = ("Airbus A320-200ceo",
                          "Airbus A320-200ceoMOD", "Airbus A320-200ceoMOD_SD", "Airbus A320-200ceoMOD_Short") # Temporary
        
        if self.dispName not in self.dispNames:
            raise NotImplementedError(f"Invalid display name '{self.dispName}'. Must be one of {self.dispNames}.")

        # Locate entries for aircraft type and engine model
        dfAircraft = pd.read_excel(DATABASE_PATH, sheet_name="Aircraft")
        dfEngines = pd.read_excel(DATABASE_PATH, sheet_name="Engines")
        aircraftEntry = dfAircraft.loc[dfAircraft["Aircraft display name"] == dispName]
        engineEntry = dfEngines.loc[dfEngines["Engine model"] == aircraftEntry["Engine model"].values[0]]

        # Extract aircraft parameters
        vsp3File = aircraftEntry["VSP3 file name"].values[0]
        self.dryCGxLoc = aircraftEntry["OE CG, m"].values[0]

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

        ### Create SUAVE vehicle
        self.suaveVehicle = vsp_read(tag=self.vspPath,
                                     units_type="SI",
                                     specified_network=None,
                                     use_scaling=True)
        self.suaveVehicle.tag = self.dispName

        ## Dry weight and weight limits
        self.suaveVehicle.mass_properties.max_takeoff = aircraftEntry["MTOW, kg"].values[0] * Units.kilogram
        self.suaveVehicle.mass_properties.operating_empty = aircraftEntry["OEW, kg"].values[0] * Units.kilogram
        self.suaveVehicle.mass_properties.max_zero_fuel = aircraftEntry["MZFW, kg"].values[0] * Units.kilogram
        self.suaveVehicle.mass_properties.max_payload = self.suaveVehicle.mass_properties.max_zero_fuel - self.suaveVehicle.mass_properties.operating_empty
        self.suaveVehicle.mass_properties.max_fuel = aircraftEntry["Max useable fuel weight, kg"].values[0] * Units.kilogram

        ## Rename wing to match SUAVE convention
        self.suaveVehicle["wings"]["main_wing"] = self.suaveVehicle["wings"]["wing"]
        del self.suaveVehicle["wings"]["wing"]
        self.suaveVehicle["wings"]["main_wing"].tag = "main_wing"

        # Use wing as reference area
        self.suaveVehicle.reference_area = self.suaveVehicle["wings"]["main_wing"].areas.reference * Units["meters**2"]

        ## Operating limits
        self.suaveVehicle.envelope.limit_load = 2.5 # A320 clean limit, assumed similar across types
        self.suaveVehicle.envelope.ultimate_load = 1.5*self.suaveVehicle.envelope.limit_load # (Assumed)
        self.cruise_mach = aircraftEntry["Cruise Mach"].values[0]

        ## Operating empty CG position
        #self.suaveVehicle.mass_properties.center_of_gravity = [[]]
        #self.suaveVehicle.mass_properties.zero_fuel_center_of_gravity =[[]]
        # Fuel avg. CG position?

        # Accessory drive and control configuration
        self.suaveVehicle.systems.control = "fully powered"
        self.suaveVehicle.systems.accessories = aircraftEntry["Market segment"].values[0]
       
        ## Engine nacelles
        if aircraftEntry['Number of engines'].values[0] != 2:
            raise ValueError(f"Baseline class only supports twin engine designs ({aircraftEntry['Number of engines'].values[0]} provided).")

        nacelle_1 = SUAVE.Components.Nacelles.Nacelle()
        nacelle_1.tag = "nacelle_1"
        nacelle_1.length = engineEntry["Nacelle length, m"].values[0] * Units.m
        nacelle_1.inlet_diameter = engineEntry["Nacelle inlet diameter, m"].values[0] * Units.m
        nacelle_1.diameter = engineEntry["Nacelle diameter, m"].values[0] * Units.m
        nacelle_1.areas.wetted = 1.1*np.pi*nacelle_1.diameter*nacelle_1.length
        nacelle_1.origin = [[aircraftEntry["Nacelle x, m"].values[0] * Units.m,
                             aircraftEntry["Nacelle y, m"].values[0] * Units.m,
                             aircraftEntry["Nacelle z, m"].values[0] * Units.m]]
        nacelle_1.flow_through = True
        nacelle_airfoil  = SUAVE.Components.Airfoils.Airfoil() 
        nacelle_airfoil.naca_4_series_airfoil = "2410"
        nacelle_1.append_airfoil(nacelle_airfoil)

        nacelle_2 = deepcopy(nacelle_1)
        nacelle_2.tag = "nacelle_2"
        nacelle_2.origin = [[aircraftEntry["Nacelle x, m"].values[0] * Units.m,
                             -aircraftEntry["Nacelle y, m"].values[0] * Units.m,
                             aircraftEntry["Nacelle z, m"].values[0] * Units.m]]
        nacelles = (nacelle_1, nacelle_2)

        for nacelle in nacelles:
            self.suaveVehicle.append_component(nacelle)

        ## Energy network (assumes twin spool turbofans only)
        self.suaveVehicle["networks"].clear()

        turbofan = SUAVE.Components.Energy.Networks.Turbofan()
        turbofan.tag = "turbofan"
        turbofan.number_of_engines = aircraftEntry["Number of engines"].values[0]
        turbofan.origin = [nacelle.origin for nacelle in nacelles]
        turbofan.working_fluid = SUAVE.Attributes.Gases.Air()
        turbofan.bypass_ratio = engineEntry["BPR"].values[0]

        # Ram effect
        ram = SUAVE.Components.Energy.Converters.Ram()
        ram.tag = "ram"
        turbofan.append(ram)

        # Inlet duct
        inletDuct = SUAVE.Components.Energy.Converters.Compression_Nozzle()
        inletDuct.tag = "inlet_nozzle"
        inletDuct.polytropic_efficiency = engineEntry["Inlet polytropic"].values[0]
        inletDuct.pressure_ratio = engineEntry["Inlet PR"].values[0]
        turbofan.append(inletDuct)

        # Fan
        fan = SUAVE.Components.Energy.Converters.Fan()
        fan.tag = "fan"
        fan.polytropic_efficiency = engineEntry["Fan polytropic"].values[0]
        fan.pressure_ratio = engineEntry["Fan PR"].values[0]
        turbofan.append(fan)

        # Low pressure compressor
        LPC = SUAVE.Components.Energy.Converters.Compressor()
        LPC.tag = "low_pressure_compressor"
        LPC.polytropic_efficiency = engineEntry["LPC polytropic"].values[0]
        LPC.pressure_ratio = engineEntry["LPC PR"].values[0]
        turbofan.append(LPC)

        # High pressure compressor
        HPC = SUAVE.Components.Energy.Converters.Compressor()
        HPC.tag = "high_pressure_compressor"
        HPC.polytropic_efficiency = engineEntry["HPC polytropic"].values[0]
        HPC.pressure_ratio = engineEntry["HPC PR"].values[0]
        turbofan.append(HPC)

        # Combustor
        combustor = SUAVE.Components.Energy.Converters.Combustor()   
        combustor.tag = "combustor"
        combustor.efficiency = engineEntry["Combustor efficiency"].values[0]
        combustor.alphac = 1.0   
        combustor.turbine_inlet_temperature = engineEntry["TET, K"].values[0]
        combustor.pressure_ratio = engineEntry["Combustor PR"].values[0]

        # Check fuel assignment
        if aircraftEntry['Fuel'].values[0] not in self.fuels.keys():
            raise ValueError(f"Class supports fuels '{self.fuels.keys()}' ({engineEntry['Combustor fuel'].values[0]} provided)")
        else:
            combustor.fuel_data = self.fuels[aircraftEntry['Fuel'].values[0]]
            if combustor.fuel_data != self.fuels[engineEntry["Combustor fuel"].values[0]]:
                print(f"\nAIRCRAFT-ENGINE FUEL MISMATCH: ENGINE {engineEntry['Engine model'].values[0]} "\
                      f"'{self.fuels[engineEntry['Combustor fuel'].values[0]].tag}', AIRCRAFT {self.dispName} '{self.fuels[aircraftEntry['Fuel'].values[0]].tag}'. "\
                      f"DEFAULTING TO AIRCRAFT FUEL.\n")
        
        # Overide hydrogen specific energy to LHV from SUAVE default of HHV
        if combustor.fuel_data == self.fuels["LH2"]:
            combustor.fuel_data.specific_energy = 119.9E6
            print("\nHYDROGEN SPECIFIC ENERGY SET TO LHV 119.9 MJ/KG FROM HHV 141.9 MJ/KG.\n")

        turbofan.append(combustor)

        # High pressure turbine
        HPT = SUAVE.Components.Energy.Converters.Turbine()
        HPT.tag = "high_pressure_turbine"
        HPT.polytropic_efficiency = engineEntry["HPT polytropic"].values[0]
        HPT.mechanical_efficiency = engineEntry["HPT mechanical"].values[0]
        turbofan.append(HPT)

        # Low pressure turbine
        LPT = SUAVE.Components.Energy.Converters.Turbine()
        LPT.tag = "low_pressure_turbine"
        LPT.polytropic_efficiency = engineEntry["LPT polytropic"].values[0]
        LPT.mechanical_efficiency = engineEntry["LPT mechanical"].values[0]
        turbofan.append(LPT)

        # Core flow nozzle
        coreNozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()
        coreNozzle.tag = "core_nozzle"
        coreNozzle.polytropic_efficiency = engineEntry["Core nozzle polytropic"].values[0]
        coreNozzle.pressure_ratio = engineEntry["Core nozzle PR"].values[0]
        turbofan.append(coreNozzle)

        # Bypass flow nozzle
        bypassNozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()
        bypassNozzle.tag = "fan_nozzle"
        bypassNozzle.polytropic_efficiency = engineEntry["Bypass nozzle polytropic"].values[0]
        bypassNozzle.pressure_ratio = engineEntry["Bypass nozzle PR"].values[0]
        turbofan.append(bypassNozzle)

        # Thrust model
        thrustModel = SUAVE.Components.Energy.Processes.Thrust()       
        thrustModel.tag = "compute_thrust" 
        thrustModel.total_design = turbofan.number_of_engines*engineEntry["Design thrust, kN"].values[0]*1000*Units.N
        turbofan.thrust = thrustModel
        
        # Design sizing conditions for mass flow
        designMach = engineEntry["Design Mach"].values[0]
        designAltitude = engineEntry["Design altitude, m"].values[0] * Units.m   
        turbofan_sizing(turbofan, designMach, designAltitude)   

        self.suaveVehicle.append_component(turbofan)  

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