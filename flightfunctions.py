import numpy as np
import random
import matplotlib.pyplot as plt

import SUAVE
from SUAVE.Core import Data
import baseline as base

A320 = base.Baseline("Airbus A320-200ceo")

def aeroSweep(vehicle, alphas, machs, altitude, deltaT=0):
    """Perform a mach/alpha sweep on a vehicle, and
    extract total lift and drag to plot a vehicle polar.

    Args:
        vehicle (SUAVE.Vehicle.Vehicle): SUAVE vehicle to perform analysis on.
        alphas (numpy.ndarray): List of angles of attack, used to vary lift coefficient.
        machs (numpy.ndarray): List of Mach numbers at which to run alpha sweep.
        altitude (float): Freestream altitude, m.
        deltaT (float, optional): ISA temperature offset. Defaults to 0.

    Returns:
        matplotlib.figure.Figure: Polar plot Figure object.
        matplotlib.axes._axes.Axes: Polar plot axes object.
        list: Aerodynamic results of type SUAVE.Core.Data.Data for each Mach.
    """

    # Results list
    results = []

    # Approximate wing areas
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted 

    aero = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aero.geometry = vehicle
    aero.initialize()

    # Vehicle needs energy networks! Botch it for now...
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    vehicle.append_component(turbofan)

    # Figure
    fig, ax = plt.subplots(dpi=200)

    # Initialise state object
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    # Compute atmospheric conditions
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo.features.planet = SUAVE.Analyses.Planets.Planet().features
    fsConds = atmo.compute_values(altitude=altitude,
                                  temperature_deviation=deltaT)
    p, T = np.array(fsConds["pressure"]), np.array(fsConds["temperature"])
    rho, mu = np.array(fsConds["density"]), np.array(fsConds["dynamic_viscosity"])
    a = np.array(fsConds["speed_of_sound"])
    
    rho = rho[:,None]
    mu = mu[:,None]
    T = T[:,None]
    p = p[:,None]
    alphas = alphas[:,None]
    
    for mach in machs:
        M = np.array([mach])
        M = M[:,None]

        state.conditions.freestream.mach_number = M
        state.conditions.freestream.density = rho
        state.conditions.freestream.dynamic_viscosity = mu
        state.conditions.freestream.temperature = T
        state.conditions.freestream.pressure = p
        state.conditions.freestream.reynolds_number = rho*a*M/mu
        state.conditions.aerodynamics.angle_of_attack = alphas

        result = aero.evaluate(state)
        print(type(result), type(fig), type(ax))
        results.append(result)

        lift = result.lift.total
        drag = result.drag.total[0]
        LD = np.divide(lift, drag)

        ax.plot(drag, lift, label=f"Mach = {mach:.3f}")
    
        ldmaxindex = np.argwhere(LD==np.max(LD))[0][0]
        ldmax = np.max(LD[ldmaxindex])
        ax.scatter(drag[ldmaxindex], lift[ldmaxindex], zorder=100,
                   marker="o", label=f"Mach {mach:.3f} max L/D = {ldmax:.2f}")

    ax.set_title(f"Polar for {vehicle.tag} at {altitude} m")
    ax.set_xlabel("Vehicle drag coefficient")
    ax.set_ylabel("Vehicle lift coefficient")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.grid()
    ax.legend()

    return fig, ax, results

fig, ax, results = aeroSweep(vehicle = A320.suaveVehicle,
                             alphas = np.linspace(-1*np.pi/180, 6*np.pi/180, 50),
                             machs = np.array(A320.cruise_mach),
                             altitude = 10000)
plt.show()

def flightMission(suaveVehicle, range, fuel, payload, climbType, cruiseAlt):

    # This needs:
    # configs_analyses
    # analyses
    # mission

    # (NOT INTERESTED IN LTO CYCLES)

    results = dict.fromkeys(["times",
                             "altitudes",
                             "densities",
                             "temperatures",
                             "pressures",
                             "TASs",
                             "CASs",
                             "climbRates",
                             "alphas",
                             "weights",
                             "lifts",
                             "drags",
                             "throtttle/thrust",
                             "highlift state"])

    return dict

def plotFlightResult(result, flags):
    # Calculate flight average MJ/RPK
    pass