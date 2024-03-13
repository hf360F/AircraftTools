import numpy as np
import pandas as pd
import scipy.interpolate, scipy.optimize
import baseline as base
import matplotlib.pyplot as plt
from copy import deepcopy
import shapely

import SUAVE

class Derivative:
    def __init__(self, baseline, modName):
        """Class defining derivatives of Baseline aircraft, including OpenVSP geometry.

        Args:
            baseline (Baseline): Instance of Baseline class.
            modName (string): Derivative modification tag.
        """

        self.baseline = deepcopy(baseline)

        self.modName = modName
        self.dispName = self.baseline.dispName + "_" + self.modName
        
        self.complete = False
        self.suaveVehicle = self.baseline.suaveVehicle
        
        ## Assume same cruise, climb and descent performance (can be overwritten)
        self.cruise_mach = self.baseline.cruise_mach

        self.climbStages = self.baseline.climbStages
        self.climbRates = self.baseline.climbRates
        self.climbEndAlts = self.baseline.climbEndAlts
        self.climbEASs = self.baseline.climbEASs
        self.climbMachs = self.baseline.climbMachs

        self.descentStages = self.baseline.descentStages
        self.descentRates = self.baseline.descentRates
        self.descentEndAlts = self.baseline.descentEndAlts
        self.descentEASs = self.baseline.descentEASs
        self.descentMachs = self.baseline.descentMachs

    def ConvertToLH2(self,
                     tankStyle = "DorsalOnly",
                     ventPressure = 1.5,
                     tankCapacity = 110,
                     tankAspect = 10,
                     insulationThickness = 0.1,
                     ullageFraction = 0.05,
                     dorsalFairingBlending="default"):
    
        # Convert energy network to liquid hydrogen
        self.suaveVehicle.networks.turbofan.combustor.fuel_data = SUAVE.Attributes.Propellants.Liquid_H2()
        self.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy = 119.6E6 # MJ/kg

        # Hydrogen state lookup from vent pressure setting
        result = scipy.optimize.minimize_scalar(lambda T: abs(psatH2(T) - ventPressure),
                                                bounds = (np.min(Ts), np.max(Ts)))
        if result.success is False:
            raise ValueError(f"Failed to lookup saturation temperature for tank vent pressure {ventPressure/1E5:.2f} bar")

        T_sat = result.x
        rho_lsat = rholH2(T_sat)
        rho_vsat = rhovH2(T_sat)
        print(f"Tank saturation temperature is {T_sat:.1f} K, liquid, vapour densities are {rho_lsat:.1f} kg/m^3, {rho_vsat:.1f} kg/m^3.")

        # Note that tank is not empty with zero hydrogen mass due to mass of vapour
        usableFuel = (1 - ullageFraction)*(rho_lsat - rho_vsat)*tankCapacity
        print(f"Usable liquid hydrogen {usableFuel:.1f} kg")
        # Include vapour mass in tank 'dry' mass

        etaGrav = 0.70 # Gravimetric efficiency
        structuralMass = usableFuel*(1/etaGrav - 1)
        emptyMass = structuralMass + (tankCapacity*rho_vsat)
        print(f"Tank empty mass is {emptyMass:.1f} kg, of which {emptyMass-structuralMass:.1f} kg is vapour\n")

        if tankStyle is "DorsalOnly":
            # Move all fuel inboard: baseline MZFW = derivative MTOW
            self.suaveVehicle.mass_properties.operating_empty = self.baseline.suaveVehicle.mass_properties.operating_empty + emptyMass
            self.suaveVehicle.mass_properties.max_takeoff = self.baseline.suaveVehicle.mass_properties.max_zero_fuel
            self.suaveVehicle.mass_properties.max_zero_fuel = self.suaveVehicle.mass_properties.max_takeoff
            self.suaveVehicle.mass_properties.max_payload = self.suaveVehicle.mass_properties.max_zero_fuel - self.suaveVehicle.mass_properties.operating_empty
            self.suaveVehicle.mass_properties.max_fuel = usableFuel

        self.complete = True

    def updateDerivGeom(tankType="dorsal", tankCapacity=10):
        # Create new derived OpenVSP
        pass

    def updateSUAVEvehicle():
        # check if createDerivGeom has been run at least once
        pass

class Tank:
    def __init__(self, tankAspect, tankVolume, etaGrav, t_ins, fidelity="basic", endType="21elliptical", show=False):
        self.tankAspect = tankAspect
        self.tankVolume = tankVolume
        self.etaGrav = etaGrav
        self.tins = t_ins
        self.fidelity = fidelity
        self.endType = endType

        # Volume factors? (expulsion efficiency, internal components)

        fidelityLevels = ("basic")
        endTypes = ("21elliptical")

        if fidelity not in fidelityLevels:
            raise ValueError(f"Tank fidelity '{fidelity}' is not implemented, must be one of '{fidelityLevels}'.")
        if endType not in endTypes:
            raise ValueError(f"End type '{endType}' is not implemented, must be one of '{endTypes}'.")

        if fidelity == "basic":
            if endType == "21elliptical":
                k_end = 0.13384 # Scaling constant: volume of 1 metre diameter end

                # Perform tank sizing calculations
                t_tot = t_ins

                # Function enforcing volume for given diameter by varying length
                newLi = lambda Di: Di/2 + 4*(tankVolume - 2*k_end*Di**3)/(np.pi*(Di**2))

                # Initial guess of internal pressure vessel dimensions
                Di = np.power(4*tankVolume/(np.pi*tankAspect), 1/3) # Cylinder with flat ends
                Li = newLi(Di)
                currentAR = (Li+2*t_tot)/(Di+2*t_tot)

                i = 0
                tol = 1E-5
                i_max = 100

                while abs(currentAR/tankAspect -1) > tol:
                    i += 1
                    if i >= i_max:
                        raise ValueError(f"Vessel design failed to converge. Iteration {i}: Di = {Di:.2f} m, Li = {Li:.2f} m")
                    Di *= (currentAR/tankAspect)**(1/3)
                    Li = newLi(Di)

                    currentAR = (Li+2*t_tot)/(Di+2*t_tot)

                if Li < Di/2:
                    raise ValueError(f"Negative cylinder length ({Li-Di/2:.2f} m): aspect ratio is unreachable.")

                print(f"Vessel design converged in {i} iterations.")
                self.Di = Di
                self.Do = self.Di + 2*t_tot
                self.endLength = self.Di/4 + t_tot
                self.Li = Li
                self.Lo = self.Li + 2*t_tot

                # Generate tank profile
                CR = self.Di*0.9045
                KR = self.Di*0.172744
                tangentAngle = np.arctan(2) # Angle at which crown and knuckle meet

                anglesC = np.linspace(tangentAngle, np.pi/2, 200)
                crownys = CR*np.cos(anglesC)
                crownxs = CR*np.sin(anglesC) - CR

                anglesK = np.linspace(np.pi, np.pi-tangentAngle, 100)
                knuckleys = self.Di/2 - KR*(1 + np.cos(anglesK))
                knucklexs = KR*np.sin(anglesK) - 0.25*self.Di

                endxs = np.concatenate((knucklexs, crownxs, np.flip(crownxs), np.flip(knucklexs)))
                endys = np.concatenate((knuckleys, crownys, -np.flip(crownys), -np.flip(knuckleys)))

                # Internal vessel co-ordinates
                tankxs = np.concatenate((-endxs, np.add(endxs, self.Li), [-endxs[0]]))
                tankxs = np.subtract(tankxs, np.min(tankxs)-t_tot) # Zero position offset
                tankys = np.concatenate((np.flip(endys), endys, [endys[-1]]))

                tankxys = []
                for x, y in zip(tankxs, tankys):
                    tankxys.append((x, y))

                internalPoly = shapely.Polygon(tankxys)
                self.vesselPoly = internalPoly.buffer(t_tot)

                if show:
                    fig, ax = plt.subplots(1, 1, dpi=200)
                    ax.plot(internalPoly.exterior.xy[0], internalPoly.exterior.xy[1], color="black")
                    ax.plot(self.vesselPoly.exterior.xy[0], self.vesselPoly.exterior.xy[1], color="red")
                    ax.set_aspect("equal")
                    ax.grid()
                    plt.show()
            
        else:
            pass
            # Define materials
                # structure: AA2219 / AA2195, composite?, SS 304/L and/or SS 316/L
                # insulation: 


# Hydrogen state data, NIST webbook
Ts = (14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25)

psats = (0.075414, 0.099458, 0.12898, 0.16475, 0.20755, 0.25821, 0.31759, 0.38656, 0.46602, 0.55688, 0.66006, 0.77651, 0.90717, 1.0530, 1.2150, 1.3941, 1.5913, 1.8075, 2.0438, 2.3012, 2.5807, 2.8833, 3.2100)

rhols = (76.969, 76.557, 76.136, 75.706, 75.264, 74.810, 74.345, 73.866, 73.375, 72.869, 72.350, 71.815, 71.265, 70.698, 70.115, 69.513, 68.893, 68.253, 67.592, 66.908, 66.199, 65.465, 64.701)

rhovs = (0.13272, 0.16960, 0.21346, 0.26503, 0.32506, 0.39431, 0.47356, 0.56360, 0.66527, 0.77942, 0.90693, 1.0488, 1.2059, 1.3793, 1.5701, 1.7796, 2.0090, 2.2598, 2.5334, 2.8315, 3.1562, 3.5095, 3.8938)
hvs = (400.22, 404.80, 409.28, 413.65, 417.91, 422.05, 426.04, 429.90, 433.59, 437.13, 440.49, 443.66, 446.64, 449.42, 451.98, 454.32, 456.43, 458.28, 459.88, 461.21, 462.24, 462.97, 463.37)
hls = (-53.622, -50.047, -46.388, -42.634, -38.777, -34.811, -30.733, -26.539, -22.224, -17.784, -13.215, -8.5111, -3.6673, 1.3228, 6.4659, 11.769, 17.241, 22.889, 28.724, 34.756, 40.997, 47.460, 54.161)
hvls = np.subtract(hvs, hls)

cpvs = np.gradient(hvs)
cpls = np.gradient(hls)

muvs = (0.64763, 0.67520, 0.70265, 0.72999, 0.75726, 0.78447, 0.81165, 0.83883, 0.86603, 0.89330, 0.92065, 0.94813, 0.97579, 1.0037, 1.0318, 1.0602, 1.0891, 1.1184, 1.1482, 1.1786, 1.2097, 1.2417, 1.2746) 
muls = (25.447, 23.918, 22.552, 21.320, 20.202, 19.181, 18.244, 17.381, 16.583, 15.843, 15.154, 14.511, 13.910, 13.346, 12.815, 12.315, 11.843, 11.396, 10.971, 10.567, 10.181, 9.8120, 9.4577)

kvs = (0.010642, 0.011117, 0.011600, 0.012089, 0.012588, 0.013095, 0.013613, 0.014142, 0.014684, 0.015239, 0.015808, 0.016393, 0.016996, 0.017616, 0.018256, 0.018918, 0.019602, 0.020311, 0.021048, 0.021814, 0.022613, 0.023448, 0.024324)
kls = (0.097633, 0.098442, 0.099195, 0.099893, 0.10053, 0.10112, 0.10165, 0.10212, 0.10253, 0.10289, 0.10319, 0.10343, 0.10361, 0.10374, 0.10381, 0.10382, 0.10378, 0.10367, 0.10351, 0.10329, 0.10300, 0.10269, 0.10233)

# Interpolators
psatH2_interp = scipy.interpolate.interp1d(Ts, psats, kind="cubic")

rhols_interp = scipy.interpolate.interp1d(Ts, rhols, kind="cubic")
rhovs_interp = scipy.interpolate.interp1d(Ts, rhovs, kind="cubic")

hvs_interp = scipy.interpolate.interp1d(Ts, hvs, kind="cubic")
hls_interp = scipy.interpolate.interp1d(Ts, hls, kind="cubic")
hvls_interp = scipy.interpolate.interp1d(Ts, hvls, kind="cubic")

cpvs_interp = scipy.interpolate.interp1d(Ts, cpvs, kind="cubic")
cpls_interp = scipy.interpolate.interp1d(Ts, cpls, kind="cubic")

kvs_interp = scipy.interpolate.interp1d(Ts, kvs, kind="cubic")
kls_interp = scipy.interpolate.interp1d(Ts, kls, kind="cubic")

muvs_interp = scipy.interpolate.interp1d(Ts, muvs, kind="cubic")
muls_interp = scipy.interpolate.interp1d(Ts, muls, kind="cubic")

# Interpolation functions
def psatH2(T):
    return psatH2_interp(T)

def rholH2(T):
    return rhols_interp(T)

def rhovH2(T):
    return rhovs_interp(T)

def hvH2(T):
    return hvs_interp(T)

def hlH2(T):
    return hls_interp(T)

def hvlH2(T):
    return hvls_interp(T)

def cpvH2(T):
    return cpvs_interp(T)

def cplH2(T):
    return cpls_interp(T)

def kvH2(T):
    return kvs_interp(T)

def klH2(T):
    return kls_interp(T)

def muvH2(T):
    return muvs_interp(T)

def mulH2(T):
    return muls_interp(T)