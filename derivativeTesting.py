import derivative as dv
import baseline as base
import openvsp as vsp
import SUAVE
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
import flightFunctions

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import shapely
from scipy import interpolate, optimize

# Tank insulation test
if False:
    AR = 5.3

    tank1 = dv.Tank(usableLH2=7600.0,
                ventPressure=1.5,
                aspectRatio=AR,
                ullageFraction=0.05,
                endGeometry="2:1elliptical",
                fidelity="AutoInsulation",
                etaGrav=0.55,
                mdot_boiloff=0.30,
                t_wall=0.005,
                show=False)


# Tank stretch test
if True:
    A320 = base.Baseline("Airbus A320-200ceo")

    # A321 to A320 Length difference 6.9 m, max dia = 3.96 m
    # Allow 10 cm servicing / support structure gap: 3.76 m max
    ARinternal = 6.9/(3.96-(2*0.10))
    Lmax = 6.9

    def tankDimRes(mLH2):
        tank1 = dv.Tank(usableLH2=mLH2,
                        ventPressure=1.5,
                        aspectRatio=ARinternal,
                        ullageFraction=0.05,
                        endGeometry="2:1elliptical",
                        fidelity="Overall",
                        etaGrav=0.65,
                        t_ins=0.15,
                        t_wall=0.005,
                        show=False,
                        verbose=False)
        return np.abs(tank1.Lo - Lmax)

    # Solve for max A321/H320 tank capacity   
    sol = optimize.minimize_scalar(tankDimRes, bounds=(0, 1E5))
    mLH2max = sol.x

    # Overide - energy parity case
    #mLH2max = 6800
    #ARinternal = 2.9

    internalTank = dv.Tank(usableLH2=mLH2max,
                           ventPressure=1.5,
                           aspectRatio=ARinternal,
                           ullageFraction=0.05,
                           endGeometry="2:1elliptical",
                           fidelity="Overall",
                           etaGrav=0.65,
                           t_ins=0.05,
                           t_wall=0.005,
                           show=True,
                           verbose=True)

    print(internalTank.Lo)

    H2_A320_internal = dv.Derivative(A320, "internalTankTest")
    H2_A320_internal.stretchFuselage(extraLength=internalTank.Lo, OEWincrease=6400) # A320 to A321
    H2_A320_internal.ConvertToLH2(tankStyle="Internal",
                                  internalTank=internalTank)


# Perform dorsal aspect ratio sweep
if False:
    #ARs = ARs = np.linspace(2, 8.8, 12)
    #Cd0s = [0.01934411754384515, 0.01913900239904206, 0.019025812708716738, 0.018932873208916626, 0.01892828339029801, 0.01892829476623625, 0.018930150060316116, 0.01893719190372869, 0.018951060225042577, 0.01897549178345485, 0.01900629535728236, 0.019035576141214785]

    #ARs = np.linspace(2, 8.8, 30)
    #CDs = [0.03487575145184732, 0.03477830588810797, 0.034706876015237934, 0.03464754398052449, 0.034599496596556686, 0.03456288357806451, 0.0345593271685431, 0.034504103625740266, 0.03445839175708866, 0.03445629422012096, 0.03445540451051007, 0.03445607512190018, 0.034457937724962306, 0.03445726749120314, 0.034457431824736144, 0.03445923414807993, 0.03446089760516834, 0.03446222620792546, 0.03446606662303679, 0.03447086945047567, 0.034476992518121245, 0.03448463728413296, 0.03449200530846532, 0.034504391016324666, 0.034515813110529446, 0.03452879392707082, 0.034540771860468014, 0.034554440365948916, 0.03456497867568509, 0.03457806504857917]

    ARs = np.linspace(2.5, 8.8, 50)
    Cd0s = [0.01916521398301831, 0.01913583021504826, 0.01910995020875785, 0.019084128323443476, 0.019059750002669257, 0.019039001648833816, 0.019020144971208623, 0.01903809445656416, 0.0189862901511414, 0.018975087076851447, 0.01896165253055532, 0.01893214974793565, 0.01893088954519744, 0.01892940724623311, 0.018928957501065397, 0.018928041728742264, 0.018928756172692813, 0.01892950025955685, 0.01892944758763743, 0.01892898377219474, 0.018928396695920596, 0.018927925193288096, 0.01892884759414337, 0.018929138265116167, 0.018929593310123418, 0.018930237628074494, 0.01893095888592183, 0.018931288649496527, 0.018932776020807056, 0.01893469781283984, 0.018937458308841405, 0.01893923949070232, 0.018942399612159027, 0.018945230612152242, 0.01894906372233592, 0.018952054204698802, 0.018956217848979787, 0.018960216172626474, 0.018968304555764978, 0.018973145045746504, 0.01897964868641883, 0.018986503838354596, 0.018992667477015907, 0.01899823011076022, 0.019004934466289914, 0.019011571666538104, 0.019016999610916806, 0.019022681327447274, 0.019029273218860044, 0.019035576141214785]

    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.plot(ARs, np.divide(Cd0s, np.min(Cd0s)), color="blue")
    ax.set_xlabel("Tank aspect ratio")
    ax.set_ylabel("Total drag coefficient $C_{d0}/C_{d0,min}$")
    ax.grid()
    plt.show()

if False:
    A320 = base.Baseline("Airbus A320-200ceo")

    # A320 dorsal tank derivative

    CDs, Cd0s = [], []
    ARs = np.linspace(2.5, 8.8, 50)

    for AR in ARs:
        H2_A320 = dv.Derivative(A320, "H2_DorsalARtest")
        tank1 = dv.Tank(usableLH2=7600.0,
                        ventPressure=1.5,
                        aspectRatio=AR,
                        ullageFraction=0.05,
                        endGeometry="2:1elliptical",
                        fidelity="Overall",
                        etaGrav=0.55,
                        t_ins=0.15,
                        t_wall=0.005,
                        show=False)
        H2_A320.ConvertToLH2(tankStyle="DorsalOnly",
                            dorsalTank=tank1, dorsalxsecNum=25, dorsalxStart=3.0,
                            Sfront=2.5, Dfront=0.15, Saft=3.5, Daft=0.35)

        # Lift coefficient
        Cl = 0.45

        # Aero analysis - need to sweep alpha to match Cl, then lookup Cd at this Cl
        results = flightFunctions.aeroSweep(vehicle = H2_A320.suaveVehicle,
                                            alphas = np.linspace(-1*np.pi/180, 6*np.pi/180, 1000),
                                            machs = np.array([H2_A320.cruise_mach]),
                                            altitude = 10668)
        closestClIndex = np.argmin(np.abs(np.subtract(results["totalLift"][0], Cl)))
        Cd = results["totalDrag"][0,closestClIndex]
        Cd0 = results["parasiticDragTotal"][0,closestClIndex]

        print(f"\n\nASPECT RATIO {AR:.2f}: Cd0 {Cd0}, Cd {Cd}\n\n")
        CDs.append(Cd)
        Cd0s.append(Cd0)

    print(CDs)
    print(Cd0s)

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