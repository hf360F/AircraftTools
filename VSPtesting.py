import openvsp as vsp
import baseline

A320_200 = baseline.Baseline(dispName = "A320-200",
                    vsp3File = "AirbusA320-200.vsp3")


# Clear API
vsp.VSPRenew()
vsp.ClearVSPModel()
vsp.DeleteAllResults()

derivativeDir = "./Derivative_Aircraft/Geometry/"

baselineDir = "./Baseline_Aircraft/Geometry/"
baselineA320 = "AirbusA320-200.vsp3"

# Import A320-200
vsp.ReadVSPFile(baselineDir+baselineA320)

# Create dorsal tank version
vsp.SetVSP3FileName("DorsalTank_"+baselineA320)

# Locate the main aircraft fuselage
vehicle = vsp.FindContainer("Vehicle", 0)
fuselage = vsp.FindGeom("Fuselage", 0)
fuselage_xsecsurf = vsp.GetXSecSurf(fuselage, 0)
fuselage_xseccnt = vsp.GetNumXSec(fuselage_xsecsurf)

# Find maximum height of fuselage
fuselageMaxHeight = 0
for i in range(fuselage_xseccnt):
    xsec = vsp.GetXSec(fuselage_xsecsurf, i)
    fuselageMaxHeight = max(fuselageMaxHeight, vsp.GetXSecHeight(xsec))

# Find all xsec parms
parms = vsp.GetXSecParmIDs(vsp.GetXSec(fuselage_xsecsurf, 1))
for parm in parms:
    #print(parm, vsp.GetParmName(parm))
    pass

# Add a fuselage for dorsal external tank as daughter of fuselage
dorsalTank = vsp.AddGeom("FUSELAGE", fuselage)
vsp.SetGeomName(dorsalTank, "DorsalTank")

# Find maximum height of tank
tank_xsecsurf = vsp.GetXSecSurf(dorsalTank, 0)
tank_xseccnt = vsp.GetNumXSec(tank_xsecsurf)
tankMaxHeight = 0
for i in range(tank_xseccnt):
    xsec = vsp.GetXSec(tank_xsecsurf, i)
    tankMaxHeight = max(tankMaxHeight, vsp.GetXSecHeight(xsec))

vsp.SetParmVal(dorsalTank, "Z_Rel_Location", "XForm", (fuselageMaxHeight+tankMaxHeight)/2)

vsp.Update()

GeomParms = vsp.GetGeomParmIDs(dorsalTank)
for Parm in GeomParms:
    #print(Parm, vsp.GetParmName(Parm), vsp.GetParmGroupName(Parm))
    pass

# Want: to find max dia of main fuselage, set tank Z to half this plus half tank dia

# Of interest:
# "X_Rel_Location", "Y_Rel_Location", "Z_Rel_Location" (and _Rotation)
    

# Move the tank to where it should be?
# Change cross section?
# Change length, etc

# Save to file
vsp.WriteVSPFile(derivativeDir+vsp.GetVSPFileName(), vsp.SET_ALL)