import derivative as dv
import baseline as base
import openvsp as vsp

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import shapely
from progress.bar import IncrementalBar

# Test cross section import
if True:
    A320 = base.Baseline("Airbus A320-200ceo")
    H2_A320_Dorsal = dv.Derivative(A320, "H2_Dorsal_01")

    print(f"\nGenerating derivative: '{H2_A320_Dorsal.dispName}'")
    ## To be moved into derivative class
    vsp.ClearVSPModel() # Remove any setup from baseline
    vsp.ReadVSPFile(H2_A320_Dorsal.baseline.vspPath)
    vsp.SetVSP3FileName(H2_A320_Dorsal.dispName)

    # Fuselage identification
    vehicle = vsp.FindContainer("Vehicle", 0)
    fuselage = vsp.FindGeom("Fuselage", 0)
    fuselage_xsecsurf = vsp.GetXSecSurf(fuselage, 0)
    fuselage_xseccnt = vsp.GetNumXSec(fuselage_xsecsurf)

    # Find all xsec parms
    parms = vsp.GetXSecParmIDs(vsp.GetXSec(fuselage_xsecsurf, 1))
    for parm in parms:
        #print(parm, vsp.GetParmName(parm))
        pass

    #fig, ax = plt.subplots(1, 1, dpi=100)
    ax = plt.figure().add_subplot(projection='3d')
 
    dorsalxstart = 5 # m, at fairing tip
    dorsalxend = 26 # m, at fairing tip
    tankDiameter = 2.5 # m
    dorsalXsecCount = 20
    tank_zoffset = 0 # m

    dorsalXsecs = []
    baselineXsecs = []
    derivativeXsecs = []

    def fairingProfile(x):
        positionOffset = 0 # m, currently broken
        TransitionScale = 0.6 # m

        startPosition = dorsalxstart - positionOffset
        endPosition = dorsalxend + positionOffset

        if x < dorsalxstart or x > dorsalxend:
            return 0
        else:
            sigmoidStart = 1/(1 + np.exp(-(x - startPosition)/TransitionScale))
            sigmoidEnd = 1/(1 + np.exp((x - endPosition)/TransitionScale))
            return np.max((0, 2*tankDiameter*(sigmoidStart + sigmoidEnd - 1.5)))
    
    if False:
        fig2, ax2 = plt.subplots(1, 1, dpi=200)
        xstest = np.linspace(0, dorsalxend+dorsalxstart, 1000)
        ystest = [fairingDiameter(x) for x in xstest]
        ax2.plot(xstest, ystest)
        ax2.axvline(dorsalxstart, color="black", linestyle="dashed")
        ax2.axvline(dorsalxend, color="black", linestyle="dashed")
        ax2.grid()
        ax2.set_aspect("equal")
        plt.show()
    
    allowableXSecTypes = ("XS_POINT", "XS_CIRCLE")

    bar = IncrementalBar("Reading baseline sections", max=fuselage_xseccnt)
    for i in range(fuselage_xseccnt):

        XSecID = vsp.GetXSec(fuselage_xsecsurf, i)
        XSecShapeNum = vsp.GetXSecShape(XSecID)
        XSEC_CRV_ENUM = XSecShapeNum

        match XSEC_CRV_ENUM:
            case -1:
                XSEC_CRV_TYPE = "XS_UNDEFINED"
            case 0:
                XSEC_CRV_TYPE = "XS_POINT"
            case 1:
                XSEC_CRV_TYPE = "XS_CIRCLE"
            case 2:
                XSEC_CRV_TYPE = "XS_ELLIPSE"
            case 3:
                XSEC_CRV_TYPE = "XS_SUPER_ELLIPSE"
            case 4:
                XSEC_CRV_TYPE = "XS_ROUNDED_RECTANGLE"
            case 5:
                XSEC_CRV_TYPE = "XS_GENERAL_FUSE"
            case 6:
                XSEC_CRV_TYPE = "XS_FILE_FUSE"
            case 7:
                XSEC_CRV_TYPE = "XS_FOUR_SERIES3"
            case 8:
                XSEC_CRV_TYPE = "XS_SIX_SERIES"
            case 9:
                XSEC_CRV_TYPE = "XS_BICONVEX"
            case 10:
                XSEC_CRV_TYPE = "XS_WEDGE"
            case 11:
                XSEC_CRV_TYPE = "XS_EDIT_CURVE"
            case 12:
                XSEC_CRV_TYPE = "XS_FILE_AIRFOIL"
            case 13:
                XSEC_CRV_TYPE = "XS_CST_AIRFOIL"
            case 14:
                XSEC_CRV_TYPE = "XS_VKT_AIRFOIL"
            case 15:
                XSEC_CRV_TYPE = "XS_FOUR_DIGIT_MOD"
            case 16:
                XSEC_CRV_TYPE = "XS_FIVE_DIGIT"
            case 17:
                XSEC_CRV_TYPE = "XS_FIVE_DIGIT_MOD"
            case 18:
                XSEC_CRV_TYPE = "XS_ONE_SIX_SERIES"
            case 19:
                XSEC_CRV_TYPE = "XS_NUM_TYPES"
            case _:
                XSEC_CRV_TYPE = None

        if XSEC_CRV_TYPE == "XS_UNDEFINED" or None:
            raise ValueError(f"Cross section {i} has XSEC_CRV_TYPE {XSEC_CRV_ENUM}, '{XSEC_CRV_TYPE}'")

        # Not really needed - can extract section profile for any general type?
        if XSEC_CRV_TYPE not in allowableXSecTypes:
            raise ValueError(f"Cross section {i} has XSEC_CRV_TYPE '{XSEC_CRV_TYPE}', which is not implemented.")

        xsec = {"ys": [],
                "zs": [],
                "xbar": None,
                "zOffset": 0,
                "baseline": True}
        xs = []

        sectionResolution = 36
        angles = np.linspace(0, 2*np.pi, sectionResolution, endpoint=False)

        if i == 0 or i == fuselage_xseccnt-1:
            vec = vsp.ComputeXSecPnt(XSecID, 0)
            xs.append(vec.x())
            xsec["ys"].append(vec.y())
            xsec["zs"].append(vec.z())
        else:
            Us = np.linspace(0, 1, sectionResolution, endpoint=False)
            for U in Us:
                vec = vsp.ComputeXSecPnt(XSecID, U)
                xs.append(vec.x())
                xsec["ys"].append(vec.y())
                xsec["zs"].append(vec.z())

        xsec["xbar"] = np.mean(xs)
        if np.max(xs) - np.min(xs) > 0.001*xsec["xbar"]:
            raise ValueError(f"Reference vehicle cross section {i} not orthogonal to x axis")

        vspXsec = vsp.GetXSec(fuselage_xsecsurf, i)
        xsec["zOffset"] = vsp.GetParmVal(vsp.GetXSecParm(vspXsec, "ZLocPercent"))

        baselineXsecs.append(xsec)
        bar.next()
    bar.finish()

    fuselageLength = np.max([xsec["xbar"] for xsec in baselineXsecs]) - np.min([xsec["xbar"] for xsec in baselineXsecs])

    # Find global max y height of baseline fuselage, used to set dorsal tank position
    z_max = np.max([np.max(xsec["zs"]) for xsec in baselineXsecs])

    # Blend the fuselage xsecs
    # Find reference cross sections either side of tank position
    xlist = [xsec["xbar"] for xsec in baselineXsecs]

    spacingExponent = 4
    mid = (dorsalxend - dorsalxstart)/2
    temp = np.linspace(-mid**spacingExponent, mid**spacingExponent, num=dorsalXsecCount)
    dorsalXsecPositions = np.sign(temp)*np.abs(temp)**(1/spacingExponent) + mid + dorsalxstart

    for i in range(dorsalXsecCount):
        # Find two adjacent baseline
        endDiffList = [x if x > 0 else np.inf for x in np.subtract(xlist, dorsalXsecPositions[i])]
        XsecIndexEnd = np.argmin(endDiffList)
        XsecIndexStart = XsecIndexEnd-1

        # New cross section
        xsec = {"ys": [],
                "zs": [],
                "xbar": None,
                "zOffset": float,
                "previousBaselineIndex": int,
                "baseline": False}

        xsec["xbar"] = dorsalXsecPositions[i]
        interpFraction = (xsec["xbar"]-baselineXsecs[XsecIndexStart]["xbar"])/(baselineXsecs[XsecIndexEnd]["xbar"] - baselineXsecs[XsecIndexStart]["xbar"])

        xsec["ys"] = baselineXsecs[XsecIndexStart]["ys"] + interpFraction*np.subtract(baselineXsecs[XsecIndexEnd]["ys"], baselineXsecs[XsecIndexStart]["ys"])
        xsec["zs"] = baselineXsecs[XsecIndexStart]["zs"] + interpFraction*np.subtract(baselineXsecs[XsecIndexEnd]["zs"], baselineXsecs[XsecIndexStart]["zs"])

        xsec["zOffset"] = baselineXsecs[XsecIndexStart]["zOffset"] + interpFraction*np.subtract(baselineXsecs[XsecIndexEnd]["zOffset"], baselineXsecs[XsecIndexStart]["zOffset"])

        xsec["previousBaselineIndex"] = int(XsecIndexStart)

        dorsalXsecs.append(xsec)

        #ax.plot(xsec["ys"], xsec["zs"], zs=xsec["xbar"], zdir="x", color="red")

    # Sort by x
    derivativeXsecs = sorted(np.concatenate((baselineXsecs, dorsalXsecs)), key=lambda d: d["xbar"])

    # Now add dorsal tank to the cross sections where relevant
    for xsec in derivativeXsecs:
        if xsec["xbar"] > dorsalxstart and xsec["xbar"] < dorsalxend:
            fairingHeight = fairingProfile(xsec["xbar"])

            tankys = fairingHeight*np.cos(angles)/2
            tankzs = z_max + fairingHeight*(1+np.sin(angles))/2 + tank_zoffset

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

        #print(f"Position {xsec['xbar']:.2f} with centre {centrey:.2f} {centrez:.2f}, index = {indexOffset}")
        if np.round(xsec['xbar'], 1) in (None, None):#(35.2, 29.3):
            yaxangles2 = np.arctan2(xsec["zs"], xsec["ys"])
            fig, ax4 = plt.subplots(1, 1)
            ax4.set_aspect("equal")
            ax4.scatter(xsec["ys"], xsec["zs"])
            ax4.scatter(xsec["ys"][0], xsec["zs"][0], color="red")

    for xsec in derivativeXsecs:
        ax.plot(xsec["ys"], xsec["zs"], zs=xsec["xbar"], zdir="x", color="black")

    ax.set_aspect("equal")
    #ax.grid()
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    #ax.set_zlabel("z")
    ax.axis("off")
    #plt.show()

    print("Derivative geometry generated, creating OpenVSP model.")

    # Save these as fuselage files
    bar = IncrementalBar("Updating sections from baseline", max=fuselage_xseccnt)
    for i in range(fuselage_xseccnt):        
        xsec = baselineXsecs[i]

        if i == 0 or i == fuselage_xseccnt-1: # Start and end should be points
            vsp.ChangeXSecShape(fuselage_xsecsurf, i, 0)
        else:
            # Convert OpenVSP shape to fuse file format
            vsp.ChangeXSecShape(fuselage_xsecsurf, i, 6)

            # Save coords into xsec fuse file
            dir = "./testxsecs/"
            name = f"xsec_buffer.fxs"
            f = open(dir+name, "w")
            line1 = f"OPENVSP_XSEC_FILE_V1\n"
            f.write(line1)
            for j in range(len(xsec["ys"])):
                f.write(f"{xsec['ys'][j]:.4f}  {xsec['zs'][j]:.4f}\n")
            f.write(f"{xsec['ys'][0]:.4f}  {xsec['zs'][0]:.4f}\n") # Loop back to start
            f.close()

            vspXsec = vsp.GetXSec(fuselage_xsecsurf, i)
            vsp.ReadFileXSec(vspXsec, dir+name)
  
        # Reapply the original Z position offset in OpenVSP
        zOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, i), "ZLocPercent")
        #print(xsec["zOffset"])
        vsp.SetParmVal(zOffId, xsec["zOffset"])
        #vsp.Update()
        bar.next()
    
    bar.finish()

    # Now insert new cross-sections
    bar = IncrementalBar("Inserting additional sections", max=len(dorsalXsecs))
    for i in range(len(dorsalXsecs)):
        xsec = dorsalXsecs[i]

        # Insert new section into OpenVSP
        vsp.InsertXSec(fuselage, xsec["previousBaselineIndex"]+i, 6)
        #vsp.Update()
        # Move to correct x position
        xOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+1), "XLocPercent")
        xPct = xsec["xbar"]/fuselageLength
        vsp.SetParmValLimits(xOffId, xPct, 0, 1)

        # z position
        zOffId = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+1), "ZLocPercent")
        vsp.SetParmVal(zOffId, xsec["zOffset"])
        
        """
        xOffId1 = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i), "XLocPercent")
        xPct1 = baselineXsecs[xsec["previousBaselineIndex"]]["xbar"]/fuselageLength
        xPct1_2 = vsp.GetParmVal(xOffId1)
        xOffId2 = vsp.GetXSecParm(vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+2), "XLocPercent")
        xPct2 = baselineXsecs[xsec["previousBaselineIndex"]+1]["xbar"]/fuselageLength
        xPct2_2 = vsp.GetParmVal(xOffId2)

        print(f"Placed xsec at {100*vsp.GetParmVal(xOffId):.2f}% (wanted {xPct*100:.2f}%). Adjacents are at {100*xPct1:.2f}% ({xPct1_2*100:.2f}%), {100*xPct2:.2f}% ({xPct2_2*100:.2f}%).")
        """

        # Save coords into xsec fuse file
        dir = "./testxsecs/"
        name = f"xsec_buffer.fxs"
        f = open(dir+name, "w")
        line1 = f"OPENVSP_XSEC_FILE_V1\n"
        f.write(line1)
        for j in range(len(xsec["ys"])):
            f.write(f"{xsec['ys'][j]:.4f}  {xsec['zs'][j]:.4f}\n")
        f.write(f"{xsec['ys'][0]:.4f}  {xsec['zs'][0]:.4f}\n") # Loop back to start
        f.close()

        vspXsec = vsp.GetXSec(fuselage_xsecsurf, xsec["previousBaselineIndex"]+i+1)
        vsp.ReadFileXSec(vspXsec, dir+name)
        bar.next()
    bar.finish()

    fname = "DERIVATIVE_TEST.vsp3"
    vsp.SetVSP3FileName(fname)
    vsp.Update()
    vsp.WriteVSPFile(fname, 0)
    print("Derivative geometry written to OpenVSP.")

# Test hydrogen EOS
if False:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=200)
    fig.suptitle("Hydrogen saturation data (NIST)")
    fig.tight_layout()
    fig, (ax3, ax4) = plt.subplots(2, 1, sharex=True, dpi=200)
    fig.suptitle("Hydrogen saturation data (NIST)")
    fig.tight_layout()
    fig, (ax5, ax6) = plt.subplots(2, 1, sharex=True, dpi=200)
    fig.suptitle("Hydrogen saturation data (NIST)")
    fig.tight_layout()

    Ts = np.linspace(14, 25, 100)

    # Curves using interpolation functions
    psats = [dv.psatH2(T) for T in Ts]

    rhols = [dv.rholH2(T) for T in Ts]
    rhovs = [dv.rhovH2(T) for T in Ts]

    hvs = [dv.hvH2(T) for T in Ts]
    hls = [dv.hlH2(T) for T in Ts]
    hvls = [dv.hvlH2(T) for T in Ts]

    cpvs = [dv.cpvH2(T) for T in Ts]
    cpls = [dv.cplH2(T) for T in Ts]

    kvs = [dv.kvH2(T) for T in Ts]
    kls = [dv.klH2(T) for T in Ts]

    muvs = [dv.muvH2(T) for T in Ts]
    muls = [dv.mulH2(T) for T in Ts]

    # Plot

    liqColour = "blue"
    vapColour = "red"

    ax1.plot(Ts, psats, color="black")
    #ax1.set_xlim(Ts[0], Ts[-1])
    ax1.set_ylim(0)
    #ax1.set_xlabel("Temperature, K")
    ax1.set_ylabel("Saturation pressure, bar")
    ax1.grid()

    ax2.plot(Ts, rhovs, label="Vapour", color=vapColour)
    ax2.plot(Ts, rhols, label="Liquid", color=liqColour)
    ax2.set_xlim(Ts[0], Ts[-1])
    ax2.set_ylim(0)
    ax2.set_xlabel("Temperature, K")
    ax2.set_ylabel("Density, kg/m$^3$")
    ax2.grid()
    ax2.legend()

    ax3.plot(Ts, hvs, label="Vapour", color=vapColour)
    ax3.plot(Ts, hls, label="Liquid", color=liqColour)
    ax3.plot(Ts, hvls, label="Vapourisation", color="purple")
    #ax3.set_xlim(Ts[0], Ts[-1])
    #ax3.set_xlabel("Temperature, K")
    ax3.set_ylabel("Enthalpy, kJ/kg")
    ax3.grid()
    ax3.legend()

    ax4.plot(Ts, cpvs, label="Vapour", color=vapColour)
    ax4.plot(Ts, cpls, label="Liquid", color=liqColour)
    ax4.set_xlim(Ts[0], Ts[-1])
    ax4.set_xlabel("Temperature, K")
    ax4.set_ylabel("Specific heat capacity $c_p$, kJ/kgK")
    ax4.grid()
    ax4.legend()

    ax5.plot(Ts, kvs, label="Vapour", color=vapColour)
    ax5.plot(Ts, kls, label="Liquid", color=liqColour)
    #ax5.set_xlim(Ts[0], Ts[-1])
    #ax5.set_xlabel("Temperature, K")
    ax5.set_ylabel("Thermal conductivity, W/mK")
    ax5.grid()
    ax5.legend()

    ax6.plot(Ts, muvs, label="Vapour", color=vapColour)
    ax6.plot(Ts, muls, label="Liquid", color=liqColour)
    ax6.set_xlim(Ts[0], Ts[-1])
    ax6.set_xlabel("Temperature, K")
    ax6.set_ylabel("Dynamic viscosity, $\mu$Pas")
    ax6.grid()
    ax6.legend()

    plt.show()