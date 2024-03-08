import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

import baseline as base
from flightFunctions import *
import SUAVE
import openvsp as vsp
from SUAVE.Core import Data, Units
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read

A320 = base.Baseline("Airbus A320-200ceo")
A320Dorsal = base.Baseline("Airbus A320-200ceoMOD")
A320DorsalSmallD = base.Baseline("Airbus A320-200ceoMOD_SD")
A320DorsalShort = base.Baseline("Airbus A320-200ceoMOD_Short")

#print(A320.climbStages, A320.climbRates, A320.climbIASs, A320.climbEndAlts)
#A320.showVehicleGeom()
#plt.show()

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

# Drag polar and breakdown
if False:
    Aircrafts = (A320, A320Dorsal)
    colours = ("red", "blue")
    i = 0

    for Aircraft in Aircrafts:
        results = aeroSweep(vehicle = Aircraft.suaveVehicle,
                            alphas = np.linspace(-1*np.pi/180, 6*np.pi/180, 100),
                            machs = np.array([0.1, 0.6, Aircraft.cruise_mach]),
                            altitude = 10668)
        
        targetMach = Aircraft.cruise_mach
        nearestMachIndex = np.argmin(np.abs(np.subtract(results["machs"], targetMach)))

        targetCl = 0.45
        Cls = results["totalLift"][nearestMachIndex]
        nearestClIndex = np.argmin(np.abs(np.subtract(Cls, targetCl)))

        mach = results["machs"][nearestMachIndex]
        Cl = results["totalLift"][nearestMachIndex][nearestClIndex]
        alpha = results['alphas'][nearestClIndex][0]

        fig, ax = plt.subplots(1, 1, dpi=200)

        dragBreakdown = (results["parasiticDragTotal"][nearestMachIndex, nearestClIndex],
                         results["inducedDrag"][nearestMachIndex, nearestClIndex],
                         results["compDrag"][nearestMachIndex, nearestClIndex],
                         results["miscDrag"][nearestMachIndex, nearestClIndex])
        dragLabels = ("Parasitic", "Induced", "Compressibility", "Miscellaneous")

        for i in range(len(results["machs"])):
            mach = results["machs"][i]

            bestLDIndex = np.argmax(results["liftToDrag"][i])
            bestLD = results["liftToDrag"][i][bestLDIndex]

            ax.scatter(results["totalDrag"][i][bestLDIndex], results["totalLift"][i][bestLDIndex],
                       label=f"M = {mach:.3f}, max L/D = {bestLD:.2f}")
            ax.plot(results["totalDrag"][i], results["totalLift"][i])

        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_title(f"{Aircraft.dispName} drag polar at {results['altitude']:.0f} m")
        ax.set_xlabel("Aircraft drag coefficient")
        ax.set_ylabel("Aircraft lift coefficient")
        ax.legend()
        ax.grid()

        fig = plt.figure(dpi=200)
        ax2 = fig.add_subplot(121)
        ax3 = fig.add_subplot(122)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0)

        def formatter1(rel_val):
            abs_val = rel_val*np.sum(dragBreakdown)/100
            pct = rel_val
            return f"{pct:.1f}% ({abs_val:.4f})"
        
        def formatter2(rel_val):
            abs_val = rel_val*dragBreakdown[0]/100
            pct = rel_val
            return f"{pct:.1f}% ({abs_val:.4f})"

        # Drag breakdown by cause
        angle = -180*dragBreakdown[0]/np.sum(dragBreakdown)
        explode = np.zeros_like(dragBreakdown)
        explode[0] = 0.1
        ax2.pie(dragBreakdown, labels=dragLabels, autopct=formatter1, startangle=angle, explode=explode)
        fig.suptitle(f"{Aircraft.dispName} drag breakdown\nM = {mach:.3f}, h = {results['altitude']:.0f} m, $C_L$ = {Cl:.3f}"\
                     r" ($\alpha$ = "+f"{180*alpha/np.pi:.2f}"+r"$^{\circ}$),"\
                     f" $C_D$ = {np.sum(dragBreakdown):.4f}")
        
        # Component breakdown for parasitic drag
        parasiticDragFull = results["parasiticDragFull"][nearestMachIndex][nearestClIndex]

        parasiticBreakdown = [parasiticDragFull["wings"]["htail"]["wing"][0,0,0],
                              parasiticDragFull["wings"]["vtail"]["wing"][0,0,0],
                              parasiticDragFull["wings"]["main_wing"]["wing"][0,0,0]]
        parasiticLabels = ["Horizontal tail",
                           "Vertical tail",
                           "Wing"]

        #print(parasiticDragFull)

        for fuselage, label in zip(parasiticDragFull["fuselages"], parasiticDragFull["fuselages"].keys()):
            fuselageDrag = 0
            for fuselageComp in fuselage.keys():
                fuselageDrag += fuselage[fuselageComp][0,0,0]

            parasiticBreakdown.append(fuselageDrag)
            parasiticLabels.append(label)

        nacelleDrag = 0
        for nacelle in parasiticDragFull["nacelles"].keys():
            nacelleDrag += parasiticDragFull["nacelles"][nacelle]["nacelle"][0,0,0]
        parasiticBreakdown.append(nacelleDrag)
        parasiticLabels.append("Nacelles")

        radius2 = 0.7
        ax3.pie(parasiticBreakdown, labels=parasiticLabels, 
                autopct=formatter2, radius=radius2, startangle=angle)

        # Draw connecting lines
        width = 0.1
        theta1, theta2 = ax2.patches[0].theta1, ax2.patches[0].theta2
        center, r = ax2.patches[0].center, ax2.patches[0].r

        # Top
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, radius2), xyB=(x, y),
                            coordsA="data", coordsB="data", axesA=ax3, axesB=ax2)
        con.set_color([0, 0, 0])
        con.set_linewidth(2)
        ax3.add_artist(con)

        # Bottom
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, -radius2), xyB=(x, y), coordsA="data",
                            coordsB="data", axesA=ax3, axesB=ax2)
        con.set_color([0, 0, 0])
        ax3.add_artist(con)
        con.set_linewidth(2)

    plt.show()

# Fixed range cruise flight; given payload, determine required fuel
if False:
    results = fixedRangeMission(Aircraft = A320Dorsal,
                                range = 3500 * Units.km,
                                payload = 5000 * Units.kg,
                                climbType = "AircraftDefined",
                                cruiseAlt = 10668 * Units.m)
    
    plot_mission(results)
    plt.show()
    
# Produce payload range diagrams
if True:
    ncols = 160
    Aircrafts = (A320, A320Dorsal)
    colours = ("red", "blue", "green")
    i = 0

    fig, ax = plt.subplots(1, 1, dpi=200)

    pbar1 = tqdm.tqdm(total=len(Aircrafts), position=1, desc=f"Building payload-range diagram", ncols=ncols)
    pbars = []
    for Aircraft in Aircrafts:
        vehicle = Aircraft.suaveVehicle

        payloads = [vehicle.mass_properties.max_payload, vehicle.mass_properties.max_payload, 0, 0]
        fuels = [0, 0, vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]
        
        payloads[2] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + fuels[2])
        fuels[1] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + payloads[1])

        ranges = np.zeros_like(payloads)

        pbars.append(tqdm.tqdm(total=len(payloads)-1, position=2+i, desc=F"  {Aircrafts[i].dispName} Flight 1", ncols=ncols))
        for j in range(1, len(payloads)):
            if fuels[j] == 0:
                ranges[j] = 0
            else:
                results =  fixedFuelMission(Aircraft = Aircraft,
                                        fuel = fuels[j],
                                        payload = payloads[j],
                                        climbType = "AircraftDefined",
                                        cruiseAlt = 10668 * Units.m)

                ranges[j] = results.segments[-1].conditions.frames.inertial.aircraft_range[-1,0]
            
            pbars[-1].update(1)
            pbars[-1].set_description(f"    {Aircrafts[i].dispName} Flight {j}")

        ax.plot(np.divide(ranges, 1000), np.divide(payloads, 1000), color=colours[i], label=Aircraft.dispName)
        i += 1
        pbar1.update(1)
    tqdm.tqdm.write("Finished building payload-range diagram.")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")
    ax.legend()
    ax.grid()

    plt.show()