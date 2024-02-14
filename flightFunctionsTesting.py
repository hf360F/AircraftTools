import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import baseline as base
from flightfunctions import *
import SUAVE
import openvsp as vsp
from SUAVE.Core import Data, Units
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read

A320 = base.Baseline("Airbus A320-200ceo")
A320Dorsal = base.Baseline("Airbus A320-200ceoMOD")

# Derivative for A320: targeting 9000 kg LH2 capacity
# ~130 m^2 tank volume
# 3 m diameter external, assumed 2.8 m internal
# Gives cylindrical length 21 m (90% of total length)
# Gravimetric efficiency is 75% = h2 / total
# Gives 3000 kg dry weight

def plot_mission(results,line_style='bo-'):
    """This function plots the results of the mission analysis and saves those results to 
    png files."""

    #print(f"REMAINING FUEL: {vehicle.mass_properties.fuel:.0f} kg")

    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style)
    
    # Plot Aerodynamic Forces 
    plot_aerodynamic_forces(results, line_style)
    
    # Plot Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results, line_style)
    
    # Drag Components
    plot_drag_components(results, line_style)
    
    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results, line_style)

    # Plot fuel use
    plot_fuel_use(results, line_style)
    
    # Plot Velocities 
    plot_aircraft_velocities(results, line_style)      
        
    return


# Polar
if False:
    fig, ax, results = aeroSweep(vehicle = A320.suaveVehicle,
                                alphas = np.linspace(-1*np.pi/180, 6*np.pi/180, 50),
                                machs = np.array([A320.cruise_mach[0]]),
                                altitude = 10000)
    plt.show()

# Fixed range cruise flight; given payload, determine required fuel
if False:
    results = fixedRangeMission(Aircraft = A320Dorsal,
                                range = 7600 * Units.km,
                                payload = 0 * Units.kg,
                                climbType = "CruiseOnly",
                                cruiseAlt = 10668 * Units.m)
    
    plot_mission(results)
    plt.show()
    
# Produce payload range diagram
if True:
    Aircrafts = (A320, A320Dorsal)
    colours = ("red", "blue")
    i = 0

    fig, ax = plt.subplots(1, 1, dpi=200)

    for Aircraft in Aircrafts:
        vehicle = Aircraft.suaveVehicle


        payloads = [vehicle.mass_properties.max_payload, vehicle.mass_properties.max_payload, 0, 0]
        fuels = [0, 0, vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]
        
        payloads[2] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + fuels[2])
        fuels[1] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + payloads[1])

        ranges = np.zeros_like(payloads)

        for j in range(1, len(payloads)):
            results = fixedFuelMission(Aircraft = Aircraft,
                                       fuel = fuels[j],
                                       payload = payloads[j],
                                       climbType = "CruiseOnly",
                                       cruiseAlt = 10668 * Units.m)

            ranges[j] = results.segments[-1].distance

        ax.plot(np.divide(ranges, 1000), np.divide(payloads, 1000), color=colours[i], label=Aircraft.dispName)
 
        i += 1

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")
    ax.legend()
    ax.grid()

    plt.show()