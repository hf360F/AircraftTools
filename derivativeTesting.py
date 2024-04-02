import derivative as dv
import baseline as base
import openvsp as vsp
import SUAVE
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import shapely
from scipy import interpolate, optimize

# Test hydrogen EOS
if True:
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