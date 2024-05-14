import numpy as np
import pandas as pd
import scipy.interpolate, scipy.optimize
import baseline as base
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import shapely
import os

import openvsp as vsp
import SUAVE
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read

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
        self.path = os.path.join("./Derivative_Aircraft/", self.dispName) 
        os.makedirs(self.path, exist_ok=True)

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

    def ConvertToLH2(self, tankStyle, 
                     dorsalTank=None, dorsalxStart=None, dorsalxsecNum=10, dorsalzOffset=0,
                     Sfront=2.0, Dfront=0.12, Saft=3.0, Daft=0.5,
                     internalTank=None):
        self.tankStyle = tankStyle
        self.dorsalTank = dorsalTank
        self.dorsalxStart = dorsalxStart
        self.dorsalxsecNum = dorsalxsecNum
        self.dorsalzOffset = dorsalzOffset
        self.internalTank = internalTank
        self.tanks = []

        tankStyles = ("DorsalOnly", "Internal", "Both")
        if self.tankStyle not in tankStyles:
            raise ValueError(f"Tank style '{self.tankStyle}' not implemented, must be one of '{tankStyles}'.")
        
        if tankStyle == "DorsalOnly":
            if self.dorsalTank is None:
                raise ValueError(f"Tank style '{self.tankStyle}' requires a Tank instance for argument 'dorsalTank'.")
            if self.dorsalxStart is None:
                raise ValueError(f"Tank style '{self.tankStyle}' requires a dorsal tank start position.")

            # Generate fairing profile
            print("Generating fairing...")
            self.dorsalFairingxs, self.dorsalFairingys = self.genDorsalFairing(self.dorsalTank, Sfront, Dfront, Saft, Daft)
            self.dorsalxEnd = self.dorsalxStart + np.max(self.dorsalFairingxs)

            # Fairing interpolation function
            fairingInterp = lambda x: np.interp(x, self.dorsalFairingxs, self.dorsalFairingys)
            self.fairingProfile = lambda x: fairingInterp(x) if x >= 0 and x <= np.max(self.dorsalFairingxs)  else 0

            print("Generated!")
            # Add fairing to OpenVSP and create suaveVehicle
            self.addDorsalGeom(dorsalxStart, dorsalxsecNum)
            self.tanks.append(dorsalTank)

        elif tankStyle in ("Internal", "Both"):
            if self.internalTank is None:
                raise ValueError(f"Tank style '{self.tankStyle}' requires a Tank instance for argument 'internalTank'.")

            self.tanks.append(internalTank)

        if tankStyle == "Both":
            if self.dorsalTank is None:
                raise ValueError(f"Tank style '{self.tankStyle}' requires a Tank instance for argument 'dorsalTank'.")
            if self.dorsalxStart is None:
                raise ValueError(f"Tank style '{self.tankStyle}' requires a dorsal tank start position.")
            
            # Generate fairing profile
            print("Generating fairing...")
            self.dorsalFairingxs, self.dorsalFairingys = self.genDorsalFairing(self.dorsalTank, Sfront, Dfront, Saft, Daft)
            self.dorsalxEnd = self.dorsalxStart + np.max(self.dorsalFairingxs)

            # Fairing interpolation function
            fairingInterp = lambda x: np.interp(x, self.dorsalFairingxs, self.dorsalFairingys)
            self.fairingProfile = lambda x: fairingInterp(x) if x >= 0 and x <= np.max(self.dorsalFairingxs)  else 0

            print("Generated!")
            # Add fairing to OpenVSP and create suaveVehicle
            self.addDorsalGeom(dorsalxStart, dorsalxsecNum)
            self.tanks.append(dorsalTank)

        # Sum contribution of all tanks to dry weight and usable fuel weight
        self.suaveVehicle.mass_properties.operating_empty = self.baseline.suaveVehicle.mass_properties.operating_empty
        self.suaveVehicle.mass_properties.max_fuel = 0
        for tank in self.tanks:
            self.suaveVehicle.mass_properties.operating_empty += tank.m_empty
            self.suaveVehicle.mass_properties.max_fuel += tank.usableLH2

        # Derived mass properties under inboard fuel assumption
        self.suaveVehicle.mass_properties.max_takeoff = self.baseline.suaveVehicle.mass_properties.max_zero_fuel # Baseline MZFW = derivative MTOW with dry wings / inboard fuel
        self.suaveVehicle.mass_properties.max_zero_fuel = self.suaveVehicle.mass_properties.max_takeoff
        self.suaveVehicle.mass_properties.max_payload = self.suaveVehicle.mass_properties.max_zero_fuel - self.suaveVehicle.mass_properties.operating_empty

        # Perform combustor LH2 conversion
        self.suaveVehicle.networks.turbofan.combustor.fuel_data = SUAVE.Attributes.Propellants.Liquid_H2()
        self.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy = 119.9E6 # MJ/kg, override to LCV from SUAVE default (HCV)

    def genDorsalFairing(self, tank, Sfront, Dfront, Saft, Daft, show=False):
        """Generate a minimum area fairing profile for a tank and add it to OpenVSP geometry.

        Args:
            tank (Tank): Instance of tank object.
            Sfront (float): Dimensionless parameter controlling front fairing angle meeting fuselage.
            Dfront (float): Dimensionless parameter relating front fairing length to tank diameters.
            Saft (float): Dimensionless parameter controlling front fairing angle meeting fuselage.
            Daft (float): Dimensionless parameter relating front fairing length to tank diameters.
            show (bool): Plot fairing and tank geometry.
        
        Returns:
            (list), (list): x and y co-ordinate arrays of fairing OML
        """

        ## Used for fairing generation
        Lfront = Dfront*tank.Do
        Laft = Daft*tank.Do

        def fairingFront(x, A1, L1=Lfront, S1=Sfront):
            def logistic1(x, A=A1, L=L1):
                return 1/(1 + np.exp(-(x-A)/L)) - 0.5
            front = (logistic1(x-S1*L1) - logistic1(A1-S1*L1))/(1 - 2*logistic1(A1-S1*L1))
            return 2*front
    
        def fairingAft(x, A2, L2=Lfront, S2=Saft):
            def logistic2(x, A=A2, L=L2):
                return 1/(1 + np.exp((x-A)/L)) - 0.5
            aft = (logistic2(x+S2*L2) - logistic2(A2+S2*L2))/(1 + 2*logistic2(A2-S2*L2))
            return 2*(aft-0.5)
        
        # Find fairing end lengths
        tol = 1E-3
        frontsoln = scipy.optimize.minimize_scalar(lambda x: np.abs(fairingFront(x, A1=0, L1=Lfront, S1=Sfront) - (1-tol)), bounds=(0, tank.Lo))
        aftsoln = scipy.optimize.minimize_scalar(lambda x: np.abs(fairingAft(x, A2=0, L2=Laft, S2=Saft) + 1 - (1-tol)), bounds=(-tank.Lo, 0))

        ## Extract one end of the tank for fairing fitting.
        tankxs = tank.vesselPoly.exterior.xy[0]
        tankys = tank.vesselPoly.exterior.xy[1]

        # Slice front end of tank, only want top half (need monotonic increasing)
        tankEndxs, tankEndys = [], []
        for i in range(len(tankxs)):
            if tankxs[i] < tank.endLength:
                if tankys[i] > 0:
                    tankEndxs.append(tankxs[i])
                    tankEndys.append(tankys[i])

        # Drop vertical and add height offset to match fairing (y = 0 is bottom of tank)
        tankEndxs = np.concatenate(([tankEndxs[0]], tankEndxs))
        tankEndys = np.concatenate(([0], np.add(tankEndys, tank.Do/2)))
        tankInterp = lambda x: np.interp(x, tankEndxs, tankEndys)

        # Function for positioning fairing ends relative to tank
        def offsetError(xOffset, fairingxs, fairingInterp, show=False):
            # Offset tank interpolator
            tankEndOffsetxs = np.add(tankEndxs, xOffset)
            tankInterpOffset = lambda x: tankInterp(x-xOffset)

            # Union of x arrays
            xcomb = [x for x in sorted(np.concatenate((tankEndOffsetxs, fairingxs))) if x <= np.min(xOffset+tank.endLength)]

            if show: # Display offset geometry
                newTankys = tankInterpOffset(xcomb)
                newFairingys = fairingInterp(xcomb)

                fig, ax = plt.subplots(1, 1, dpi=200)
                ax.set_aspect("equal")
                ax.grid()
                ax.plot(xcomb, newFairingys, label="Fairing")
                ax.plot(xcomb, newTankys, label="Tank")
                ax.legend()
                plt.show()

            # Check fairing is always higher
            maxHeightDiff = 0
            for x in xcomb:
                if tankInterp(x) > fairingInterp(x):
                    maxHeightDiff = np.max((maxHeightDiff, tankInterpOffset(x)-fairingInterp(x)))
            if maxHeightDiff > 0:
                return maxHeightDiff

            # Minimise height difference
            crossPosSoln = scipy.optimize.minimize_scalar(lambda x: np.abs(tankInterpOffset(x) - fairingInterp(x)),
                                                          bounds=(np.max((np.min(xcomb), xOffset)), np.max(xcomb)))
            
            return crossPosSoln.fun

        # Function for wetted area figure of merit and fairing profile generation
        def fairingArea(oversizeFactor, returnGeom=False):
            maxHeight = tank.Do*oversizeFactor

            fairingResolution = 100
            # Generate fairing profiles in this range. Note aft end is mirrored.
            frontxs = np.linspace(0, frontsoln.x, fairingResolution)
            frontys = [maxHeight*fairingFront(x, A1=0, L1=Lfront, S1=Sfront) for x in frontxs]
            aftxs = np.linspace(aftsoln.x, 0, fairingResolution)
            aftys = np.flip([maxHeight*(1+fairingAft(x, A2=0, L2=Laft, S2=Laft)) for x in aftxs])
            aftxs = np.subtract(aftxs, aftxs[0])

            # Geometry interpolation functions
            fairingFrontInterp = lambda x: np.interp(x, frontxs, frontys)
            fairingAftInterp = lambda x: np.interp(x, aftxs, aftys)

            tol = 1E-3
            frontOffsetSoln = scipy.optimize.minimize_scalar(offsetError, args=(frontxs, fairingFrontInterp),
                                                             bounds=(0, np.max(frontxs)), tol=tol)
            frontOffset = frontOffsetSoln.x

            aftOffsetSoln = scipy.optimize.minimize_scalar(offsetError, args=(aftxs, fairingAftInterp),
                                                           bounds=(0, np.max(aftxs)), tol=tol)
            aftOffset = aftOffsetSoln.x

            xs = np.linspace(-frontOffset, tank.Lo+aftOffset+frontOffset, 500)
            ys = [maxHeight*(fairingFront(x, A1=-frontOffset, L1=Lfront, S1=Sfront)
                             +fairingAft(x, A2=tank.Lo+aftOffset+frontOffset, L2=Laft, S2=Saft)) for x in xs]

            wettedAreaProxy = np.trapz(np.power(ys, 2), xs)
            #print(f"Oversize {oversizeFactor:.3f}: Forward offset {frontOffset:.3f} m, aft offset {aftOffset:.3f} m, area FOM {wettedAreaProxy:.1f}")
            #print(f"(Iteration of area loop, subiterations {frontOffsetSoln.nit} + {aftOffsetSoln.nit})")

            if returnGeom:
                return np.add(xs, frontOffset), ys, frontOffset, aftOffset
            else:
                return wettedAreaProxy
            
        bestFairingSoln = scipy.optimize.minimize_scalar(fairingArea, bounds=(1, 2), tol=1E-3)
        xs, ys, frontOffset, aftOffset = fairingArea(bestFairingSoln.x, returnGeom=True)

        if show:
            fig, ax = plt.subplots(1, 1, dpi=200)
            ax.plot(np.add(tankxs, frontOffset), np.add(tankys, tank.Do/2), color="red")
            ax.plot(xs, ys, color="blue")
            ax.set_aspect("equal")
            ax.grid()
            ax.set_xlabel("x, $m$")
            ax.set_ylabel("z, $m$")
            plt.show()

        return xs, ys

    def addDorsalGeom(self, dorsalxStart, dorsalxsecNum, show=False):
        print("Adding fairing to OpenVSP")

        # VSP reset
        vsp.ClearVSPModel()
        if self.tankStyle == "Both":
            vsp.ReadVSPFile(self.path+"/"+self.dispName+".vsp3") # Internal tank may use fuselage stretch
        else:
            vsp.ReadVSPFile(self.baseline.vspPath)
        vsp.SetVSP3FileName(self.dispName)

        # Vehicle and fuselage container identification
        vehicle = vsp.FindContainer("Vehicle", 0)
        fuselage = vsp.FindGeom("Fuselage", 0)
        fuselage_xsecsurf = vsp.GetXSecSurf(fuselage, 0)
        fuselage_xseccnt = vsp.GetNumXSec(fuselage_xsecsurf)

        baselineXsecs, dorsalXsecs = [], []
        # Read baseline sections        
        ncols = 80
        pbar = tqdm(total=fuselage_xseccnt, desc="Reading baseline fuselage", ncols=ncols)
        # Read baseline geometry cross-sections
        for i in range(fuselage_xseccnt):
            xsec = {"ys": [],
                    "zs": [],
                    "xbar": None,
                    "zOffset": 0,
                    "baseline": True}
            xs = []

            XSecID = vsp.GetXSec(fuselage_xsecsurf, i)

            sectionResolution = 100
            angles = np.linspace(0, 2*np.pi, sectionResolution, endpoint=False)

            if i == 0 or i == fuselage_xseccnt-1: # Points at fuselage ends
                vec = vsp.ComputeXSecPnt(XSecID, 0)
                xs.append(vec.x())
                xsec["ys"].append(vec.y())
                xsec["zs"].append(vec.z())
            else: # Else read entire curve
                Us = np.linspace(0, 1, sectionResolution, endpoint=False)
                for U in Us:
                    vec = vsp.ComputeXSecPnt(XSecID, U)
                    xs.append(vec.x())
                    xsec["ys"].append(vec.y())
                    xsec["zs"].append(vec.z())

            xsec["xbar"] = np.mean(xs)
            if np.max(xs) - np.min(xs) > 0.001*xsec["xbar"]:
                raise ValueError(f"Reference vehicle cross section {i} not orthogonal to x axis")

            xsec["zOffset"] = vsp.GetParmVal(vsp.GetXSecParm(XSecID, "ZLocPercent"))
            baselineXsecs.append(xsec)
            pbar.update(1)

        # Fuselage dimensions
        fuselageLength = np.max([xsec["xbar"] for xsec in baselineXsecs]) - np.min([xsec["xbar"] for xsec in baselineXsecs])
        if self.dorsalxStart + np.max(self.dorsalFairingxs) > np.max([xsec["xbar"] for xsec in baselineXsecs]):
            raise ValueError(f"Fairing ends at x = {self.dorsalxStart + np.max(self.dorsalFairingxs):.2f} m,"+\
                             f"fuselage ends at x = {np.max([xsec['xbar'] for xsec in baselineXsecs]):.2f} m. Try smaller aspect ratio.")

        z_max = np.max([np.max(xsec["zs"]) for xsec in baselineXsecs])

        # Generate dorsal cross locations, use tighter spacing at ends
        spacingExponent = 1.5 # Higher exponent increases density at ends of distribution
        mid = (self.dorsalxEnd - self.dorsalxStart)/2
        temp = np.linspace(-mid**spacingExponent, mid**spacingExponent, num=self.dorsalxsecNum)
        dorsalXsecPositions = np.sign(temp)*np.abs(temp)**(1/spacingExponent) + mid + self.dorsalxStart

        xlist = [xsec["xbar"] for xsec in baselineXsecs]

        if show:
            ax = plt.figure().add_subplot(projection='3d')

        # Interpolate baseline fuselage sections at new section locations
        for i in range(self.dorsalxsecNum):
            # Find two adjacent baseline sections
            endDiffList = [x if x > 0 else np.inf for x in np.subtract(xlist, dorsalXsecPositions[i])]
            XsecIndexEnd = np.argmin(endDiffList)
            XsecIndexStart = XsecIndexEnd-1

            xsec = {"ys": [],
                    "zs": [],
                    "xbar": None,
                    "zOffset": float,
                    "previousBaselineIndex": int,
                    "baseline": False}

            # Perform linear interpolation on properties between adjacent baseline sections
            xsec["xbar"] = dorsalXsecPositions[i]
            interpFraction = (xsec["xbar"]-baselineXsecs[XsecIndexStart]["xbar"])/(baselineXsecs[XsecIndexEnd]["xbar"] - baselineXsecs[XsecIndexStart]["xbar"])

            xsec["ys"] = baselineXsecs[XsecIndexStart]["ys"] + interpFraction*np.subtract(baselineXsecs[XsecIndexEnd]["ys"], baselineXsecs[XsecIndexStart]["ys"])
            xsec["zs"] = baselineXsecs[XsecIndexStart]["zs"] + interpFraction*np.subtract(baselineXsecs[XsecIndexEnd]["zs"], baselineXsecs[XsecIndexStart]["zs"])

            xsec["zOffset"] = baselineXsecs[XsecIndexStart]["zOffset"] + interpFraction*np.subtract(baselineXsecs[XsecIndexEnd]["zOffset"], baselineXsecs[XsecIndexStart]["zOffset"])

            xsec["previousBaselineIndex"] = int(XsecIndexStart)

            dorsalXsecs.append(xsec)

        # Combined list, in x-increasing order
        derivativeXsecs = sorted(np.concatenate((baselineXsecs, dorsalXsecs)), key=lambda d: d["xbar"])

        # Add fairing profile to relevant cross sections
        for xsec in derivativeXsecs:
            if xsec["xbar"] > self.dorsalxStart and xsec["xbar"] < self.dorsalxEnd:
                fairingHeight = self.fairingProfile(xsec["xbar"] - self.dorsalxStart)

                tankys = fairingHeight*np.cos(angles)/2
                tankzs = z_max + fairingHeight*(1+np.sin(angles))/2 + self.dorsalzOffset

                combinedys = np.concatenate((xsec["ys"], tankys))
                combinedzs = np.concatenate((xsec["zs"], tankzs))

                zippedCoords = []
                for x, y in zip(combinedys, combinedzs):
                    zippedCoords.append((x, y))

                # Extract outer hull
                poly = shapely.Polygon(zippedCoords)
                outery, outerz = poly.convex_hull.exterior.xy

                # Update xsec
                xsec["ys"] = outery
                xsec["zs"] = outerz
            
            # Zshift
            xsec["zs"] = np.subtract(xsec["zs"], fuselageLength*xsec["zOffset"])

            # Now find the point with smallest +y axis angle. Centre offset is needed.
            if len(xsec["ys"]) > 1: # Check not a point
                zippedCoords = []
                for x, y in zip(xsec["ys"], xsec["zs"]):
                    zippedCoords.append((x, y))
                poly2 = shapely.Polygon(zippedCoords)
                centrey, centrez = shapely.centroid(poly2).x, shapely.centroid(poly2).y
            else:
                centrey, centrez = 0, 0

            #centrey, centrez = 0,0
            yaxangles = np.arctan2(np.subtract(xsec["zs"], centrez), np.subtract(xsec["ys"], centrey))
            indexOffset = -np.argmin(np.abs(yaxangles))
            # Roll so closest to +y is at start
            xsec["ys"] = np.roll(xsec["ys"], indexOffset)
            xsec["zs"] = np.roll(xsec["zs"], indexOffset)

            if show:
                ax.plot(xsec["ys"], xsec["zs"], zs=xsec["xbar"], zdir="x", color="red")

        pbar = tqdm(total=fuselage_xseccnt, desc="Updating OpenVSP fuselage (baseline)", ncols=ncols)

        for i in range(fuselage_xseccnt):        
            xsec = baselineXsecs[i]
            if i == 0 or i == fuselage_xseccnt-1: # Start and end should be points
                vsp.ChangeXSecShape(fuselage_xsecsurf, i, 0)
            else:
                # Convert OpenVSP shape to fuse file format
                vsp.ChangeXSecShape(fuselage_xsecsurf, i, 6)

                # Save coords into xsec fuse file
                name = f"/xsec_buffer.fxs"
                f = open(self.path+name, "w")
                line1 = f"OPENVSP_XSEC_FILE_V1\n"
                f.write(line1)
                for j in range(len(xsec["ys"])):
                    f.write(f"{xsec['ys'][j]:.4f}  {xsec['zs'][j]:.4f}\n")
                f.write(f"{xsec['ys'][0]:.4f}  {xsec['zs'][0]:.4f}\n") # Loop back to start
                f.close()

                vspXsec = vsp.GetXSec(fuselage_xsecsurf, i)
                vsp.ReadFileXSec(vspXsec, self.path+name)
    
            # Reapply the original Z position offset in OpenVSP
            zOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, i), "ZLocPercent")
            vsp.SetParmVal(zOffId, xsec["zOffset"])
            pbar.update(1)

        # Now insert new cross-sections
        pbar = tqdm(total=self.dorsalxsecNum, desc="Inserting new geometry", ncols=ncols)
        for i in range(len(dorsalXsecs)):
            xsec = dorsalXsecs[i]

            # Insert new section into OpenVSP
            vsp.InsertXSec(fuselage, xsec["previousBaselineIndex"]+i, 6)
            # Move to correct x position
            xOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+1), "XLocPercent")
            xPct = xsec["xbar"]/fuselageLength
            vsp.SetParmValLimits(xOffId, xPct, 0, 1)

            # z position
            zOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+1), "ZLocPercent")
            vsp.SetParmVal(zOffId, xsec["zOffset"])
            
            # Save coords into xsec fuse file
            name = f"/xsec_buffer.fxs"
            f = open(self.path+name, "w")
            line1 = f"OPENVSP_XSEC_FILE_V1\n"
            f.write(line1)
            for j in range(len(xsec["ys"])):
                f.write(f"{xsec['ys'][j]:.4f}  {xsec['zs'][j]:.4f}\n")
            f.write(f"{xsec['ys'][0]:.4f}  {xsec['zs'][0]:.4f}\n") # Loop back to start
            f.close()

            vspXsec = vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+1)
            vsp.ReadFileXSec(vspXsec, self.path+name)
            pbar.update()

        fname = self.dispName+".vsp3"
        vsp.SetVSP3FileName(fname)
        vsp.Update()
        vsp.WriteVSPFile(vsp.GetVSPFileName(), 0)
        os.replace(fname, self.path+"/"+fname)
        self.vspfname = fname
        print("Derivative geometry written to OpenVSP.")

        # Update vehicle
        vsp.ClearVSPModel()
        self.suaveVehicle.fuselages = vsp_read(tag=self.path+"/"+fname,
                                               units_type="SI",
                                               specified_network=None,
                                               use_scaling=True).fuselages
        
        if show:
            ax.set_aspect("equal")
            plt.show()

    def addInternalTank(self, stretch=True):
        pass

    def stretchFuselage(self, extraLength, OEWincrease):
        # Apply OEW increase
        self.suaveVehicle.mass_properties.operating_empty += OEWincrease

        # VSP reset
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(self.baseline.vspPath)
        vsp.SetVSP3FileName(self.dispName)

        # Vehicle and fuselage container identification
        vehicle = vsp.FindContainer("Vehicle", 0)
        wing = vsp.FindGeom("Wing", 0)
        fuselage = vsp.FindGeom("Fuselage", 0)
        fuselage_xsecsurf = vsp.GetXSecSurf(fuselage, 0)
        fuselage_xseccnt = vsp.GetNumXSec(fuselage_xsecsurf)

        sectionResolution = 100
        baselineXsecs = []
        # Read sections to find "largest" section      
        for i in range(fuselage_xseccnt):
            XSecID = vsp.GetXSec(fuselage_xsecsurf, i)
            xsec = {"index": i,
                    "ID": XSecID,
                    "width": vsp.GetXSecWidth(XSecID),
                    "height": vsp.GetXSecHeight(XSecID),
                    "xbar": 0}

            xs = []
            if i == 0 or i == fuselage_xseccnt-1: # Points at fuselage ends
                vec = vsp.ComputeXSecPnt(XSecID, 0)
                xs.append(vec.x())
            else: # Else read entire curve
                Us = np.linspace(0, 1, sectionResolution, endpoint=False)
                for U in Us:
                    vec = vsp.ComputeXSecPnt(XSecID, U)
                    xs.append(vec.x())

            xsec["xbar"] = np.mean(xs)
            if np.max(xs) - np.min(xs) > 0.001*xsec["xbar"]:
                raise ValueError(f"Reference vehicle cross section {i} not orthogonal to x axis")

            baselineXsecs.append(xsec)
        
        # Fuselage lengths
        baselineLength = np.max([xsec["xbar"] for xsec in baselineXsecs]) - np.min([xsec["xbar"] for xsec in baselineXsecs])
        derivativeLength = baselineLength + extraLength
        # Update fuselage length
        LengthID = vsp.GetParm(fuselage, "Length", "Design")
        vsp.SetParmVal(LengthID, derivativeLength)

        # Largest sections assumed to be section with largest min(width, height) 
        smallestDims = [np.min((xsec["width"], xsec["height"])) for xsec in baselineXsecs]
        largestxsecs = (baselineXsecs[np.argsort(smallestDims)[-1]], baselineXsecs[np.argsort(smallestDims)[-2]])

        maxDia = smallestDims[largestxsecs[0]["index"]] # Larger of the two biggest xsecs
        xCentre = (largestxsecs[0]["xbar"]+largestxsecs[1]["xbar"])/2

        # Move all xsecs aft of xCentre forward by extraLength
        derivativeXsecs = deepcopy(baselineXsecs)
        for xsec in derivativeXsecs:
            if xsec["xbar"] < xCentre:
                multiplier = 0
            else:
                multiplier = 1
            xsec["xbar"] += multiplier*extraLength

            # Update xsec position
            xOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["index"]), "XLocPercent")
            xPct = xsec["xbar"]/derivativeLength
            vsp.SetParmValLimits(xOffId, xPct, 0, 1)

        # Shift entire fuselage forward to maintain relative position of other components (tail)
        FuseXPosID = vsp.GetParm(fuselage, "X_Location", "XForm")
        oldXPos = vsp.GetParmVal(FuseXPosID)
        vsp.SetParmVal(FuseXPosID, oldXPos-extraLength)
        # Shift main wing forward by half of extraLength ()
        WingXPosID = vsp.GetParm(wing, "X_Location", "XForm")
        oldXPos = vsp.GetParmVal(WingXPosID)
        vsp.SetParmVal(WingXPosID, oldXPos-extraLength/2)

        # Save geometry
        fname = self.dispName+".vsp3"
        vsp.SetVSP3FileName(fname)
        vsp.Update()
        vsp.WriteVSPFile(vsp.GetVSPFileName(), 0)
        os.replace(fname, self.path+"/"+fname)
        self.vspfname = fname
        print("Derivative geometry written to OpenVSP.")

        # Update vehicle
        vsp.ClearVSPModel()
        self.suaveVehicle.fuselages = vsp_read(tag=self.path+"/"+fname,
                                               units_type="SI",
                                               specified_network=None,
                                               use_scaling=True).fuselages

class Tank:
    def __init__(self,
                 usableLH2,
                 aspectRatio,
                 ventPressure,
                 ullageFraction,
                 endGeometry,
                 fidelity="Manual",
                 etaGrav=None,
                 mdot_boiloff=None,
                 t_ins=None,
                 t_wall=None,
                 sigma_ywall=None,
                 lambda_wall=None,
                 rho_wall=None,
                 wallSafetyFactor=None,
                 lambda_ins=None,
                 rho_ins=None,
                 show=False,
                 verbose=True):

        self.usableLH2 = usableLH2
        self.aspectRatio = aspectRatio
        self.ventPressure = ventPressure
        self.ullageFraction = ullageFraction
        self.endGeometry = endGeometry
        self.fidelity = fidelity
        self.etaGrav = etaGrav
        self.mdot_boiloff = mdot_boiloff
        self.t_ins = t_ins
        self.t_wall = t_wall
        self.sigma_ywall = sigma_ywall
        self.lambda_wall = lambda_wall
        self.rho_wall = rho_wall
        self.wallSafetyFactor = wallSafetyFactor
        self.rho_ins = rho_ins
        self.lambda_ins = lambda_ins
        self.show = show
        self.verbose = verbose

        fidelityLevels = ("Overall", "Component", "AutoInsulation")
        endTypes = ("2:1elliptical")

        if self.fidelity not in fidelityLevels:
            raise ValueError(f"Tank fidelity '{self.fidelity}' is not implemented, must be one of '{fidelityLevels}'.")
        if self.endGeometry not in endTypes:
            raise ValueError(f"End type '{self.endGeometry}' is not implemented, must be one of '{endTypes}'.")

        if self.fidelity == "Overall" or "AutoInsulation":
            if self.fidelity == "Overall":
                reqdArgs = {"etaGrav": self.etaGrav, "t_ins": self.t_ins, "t_wall": self.t_wall}
            elif self.fidelity == "AutoInsulation":
                reqdArgs = {"etaGrav": self.etaGrav, "t_wall": self.t_wall, "mdot_boiloff": self.mdot_boiloff}

            for key in reqdArgs:
                if reqdArgs[key] == None:
                    raise ValueError(f"Fidelity level '{self.fidelity}' missing required argument '{key}'")

        self.T_sat = TsatH2(self.ventPressure)

        if self.fidelity == "AutoInsulation":
            deltahvap = hvlH2(self.T_sat)
            Qmax = self.deltahvap*self.mdot_boiloff
            self.t_ins = 0 # Initial guess

            deltaTwall = (273.15+30 - self.T_sat) # K

            # Using EPS foam
            lambda_ins = 0.026 # W/mK
            rho_ins = 50 # kg/m^3

        if self.fidelity == "Overall":
            if self.endGeometry == "2:1elliptical":
                k_end = 0.13384 # Scaling constant: volume of 1 metre diameter end

                # Determine required pressure vessel volume using saturated phase densities
                self.T_sat = scipy.optimize.minimize_scalar(lambda T: np.abs(psatH2(T) - ventPressure),
                                                            bounds = (np.min(Ts), np.max(Ts))).x
                self.rho_l = rholH2(self.T_sat)
                self.rho_v = rhovH2(self.T_sat)

                self.tankCapacity = self.usableLH2/((1 - self.ullageFraction)*(self.rho_l - self.rho_v))
                if verbose:
                    print(f"Usable LH2 {self.usableLH2:.1f} kg requires tank volume {self.tankCapacity:.1f} m^3 (effective density {self.usableLH2/self.tankCapacity:.1f} kg/m^3)")

                self.m_struct = self.usableLH2*(1/self.etaGrav - 1)
                self.m_empty = self.m_struct + self.tankCapacity*self.rho_v
                if verbose:
                    print(f"Vessel structural weight {self.m_struct:.1f} kg, empty weight {self.m_empty:.1f} kg including vapour")

                # Total wall thickness
                self.t_tot = self.t_ins + self.t_wall

                # Function enforcing volume for given ID by varying length
                newLi = lambda Di: Di/2 + 4*(self.tankCapacity - 2*k_end*Di**3)/(np.pi*(Di**2))

                # Error in aspect ratio for given ID
                ARresidual = lambda Di: np.abs((newLi(Di)+2*self.t_tot)/(Di+2*self.t_tot) - self.aspectRatio)

                soln = scipy.optimize.minimize_scalar(ARresidual)
                self.Di = soln.x
                self.Li = newLi(self.Di)

                self.Awet = 0

                if fidelity == "AutoInsulation":
                    t_insold = self.t_ins

                    self.Awet = 0
                    self.t_ins = deltaTwall*self.Awet*lambda_ins/Qmax
                    while np.abs((self.t_ins-t_insold)/t_insold) > 1E-2:
                        pass
                        # loop 

                if self.Li < self.Di/2:
                    raise ValueError(f"Negative cylinder length ({self.Li-self.Di/2:.2f} m): aspect ratio is unreachable.")

                self.Do = self.Di + 2*self.t_tot
                self.Lo = self.Li + 2*self.t_tot
                self.endLength = self.Di/4 + self.t_tot

                if verbose:
                    print(f"Vessel geometry converged with external diameter {self.Do:.2f} m, length {self.Lo:.2f} m")

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
                tankxs = np.subtract(tankxs, np.min(tankxs)-self.t_tot) # Zero position offset
                tankys = np.concatenate((np.flip(endys), endys, [endys[-1]]))

                tankxys = []
                for x, y in zip(tankxs, tankys):
                    tankxys.append((x, y))

                internalPoly = shapely.Polygon(tankxys)
                self.vesselPoly = internalPoly.buffer(self.t_tot)

                if show:
                    fig, ax = plt.subplots(1, 1, dpi=200)
                    ax.plot(internalPoly.exterior.xy[0], internalPoly.exterior.xy[1], color="black")
                    ax.plot(self.vesselPoly.exterior.xy[0], self.vesselPoly.exterior.xy[1], color="red")
                    ax.set_aspect("equal")
                    ax.grid()
                    plt.show()
            
        else:
            raise ValueError("Fidelity level implementation not complete!") 


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
TsatH2_interp = scipy.interpolate.interp1d(psats, Ts, kind="cubic")

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

def TsatH2(p):
    return TsatH2_interp(p)

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