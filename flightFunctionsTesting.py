import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

import baseline as base
import derivative as dv
from flightFunctions import *
import SUAVE
import openvsp as vsp
from SUAVE.Core import Data, Units
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
from copy import deepcopy

A320 = base.Baseline("Airbus A320-200ceo")

if False: # Define derivatives
    A321_deltaOEW = 6400-1600 # kg, difference from A320 (200 kg / row x 8 rows)
    A321_deltaMZFW = 11300 # kg, difference from A320
    A321_stretchLength = 6.9 # m, from A320

    # A320 dorsal tank derivative - max ferry range
    H2_A320_stockL = dv.Derivative(A320, "H2_MZFW+0_lrg")
    tankL = dv.Tank(usableLH2=7600.0,
                    ventPressure=1.5,
                    aspectRatio=6.0,
                    ullageFraction=0.05,
                    endGeometry="2:1elliptical",
                    fidelity="Overall",
                    etaGrav=0.55,
                    t_ins=0.15,
                    t_wall=0.005,
                    show=False)
    H2_A320_stockL.ConvertToLH2(tankStyle="DorsalOnly",
                        dorsalTank=tankL, dorsalxsecNum=10, dorsalxStart=3.0,
                        Sfront=2.5, Dfront=0.15, Saft=3.5, Daft=0.35)

    # Uprated MZFW to A321 value
    H2_A320_MZFWplusL = deepcopy(H2_A320_stockL)
    H2_A320_MZFWplusL.modName = "A321_MZFW_dors"
    H2_A320_MZFWplusL.dispName = f"A320, {tankL.usableLH2/1000:.1f}T LH2 dorsal w/ A321 MZFW"
    H2_A320_MZFWplusL.suaveVehicle.mass_properties.max_zero_fuel += A321_deltaMZFW
    H2_A320_MZFWplusL.suaveVehicle.mass_properties.max_takeoff += A321_deltaMZFW
    #H2_A320_MZFWplusL.suaveVehicle.mass_properties.operating_empty += A321_deltaOEW 
    H2_A320_MZFWplusL.suaveVehicle.mass_properties.max_payload = H2_A320_MZFWplusL.suaveVehicle.mass_properties.max_zero_fuel - H2_A320_MZFWplusL.suaveVehicle.mass_properties.operating_empty

    # A320 dorsal tank derivative - most common range
    H2_A320_stockS = dv.Derivative(A320, "H2_MZFW+0_sml")
    tankS = dv.Tank(usableLH2=4000.0,
                    ventPressure=1.5,
                    aspectRatio=3.3,
                    ullageFraction=0.05,
                    endGeometry="2:1elliptical",
                    fidelity="Overall",
                    etaGrav=0.55,
                    t_ins=0.15,
                    t_wall=0.005,
                    show=False)
    H2_A320_stockS.ConvertToLH2(tankStyle="DorsalOnly",
                        dorsalTank=tankS, dorsalxsecNum=10, dorsalxStart=3.0,
                        Sfront=2.5, Dfront=0.15, Saft=3.5, Daft=0.35)

    # Uprated MZFW to A321 value
    H2_A320_MZFWplusS = deepcopy(H2_A320_stockS)
    H2_A320_MZFWplusS.modName = "A321_MZFW_dors"
    H2_A320_MZFWplusS.dispName = f"A320, {tankS.usableLH2/1000:.1f}T LH2 dorsal w/ A321 MZFW"
    H2_A320_MZFWplusS.suaveVehicle.mass_properties.max_zero_fuel += A321_deltaMZFW
    H2_A320_MZFWplusS.suaveVehicle.mass_properties.max_takeoff += A321_deltaMZFW
    #H2_A320_MZFWplusS.suaveVehicle.mass_properties.operating_empty += A321_deltaOEW 
    H2_A320_MZFWplusS.suaveVehicle.mass_properties.max_payload = H2_A320_MZFWplusS.suaveVehicle.mass_properties.max_zero_fuel - H2_A320_MZFWplusS.suaveVehicle.mass_properties.operating_empty


    """
    # A320 with internal tank
    internalTank = dv.Tank(usableLH2=3610,
                            ventPressure=1.5,
                            aspectRatio=1.83,
                            ullageFraction=0.05,
                            endGeometry="2:1elliptical",
                            fidelity="Overall",
                            etaGrav=0.65,
                            t_ins=0.15,
                            t_wall=0.005,
                            show=False,
                            verbose=True)
    H2_A321_int = dv.Derivative(A320, "A321_int")
    H2_A321_int.stretchFuselage(extraLength=6.9, OEWincrease=A321_deltaOEW) # A320 to A321
    H2_A321_int.suaveVehicle.mass_properties.max_zero_fuel += A321_deltaMZFW
    H2_A321_int.ConvertToLH2(tankStyle="Internal",
                            internalTank=internalTank)


    totalLH2 = tankL.usableLH2
    dorsTank2 = dv.Tank(usableLH2 = tankL.usableLH2 - internalTank.usableLH2,
                        ventPressure=1.5,
                        aspectRatio=12,
                        ullageFraction=0.05,
                        endGeometry="2:1elliptical",
                        fidelity="Overall",
                        etaGrav = tankL.etaGrav,
                        t_ins=0.15,
                        t_wall=0.005,
                        show=False,
                        verbose=True)

    H2_A321_intdors = dv.Derivative(A320, "A321_int+dors")
    H2_A321_intdors.stretchFuselage(extraLength=A321_stretchLength, OEWincrease=A321_deltaOEW) # Stretch to 321 size
    H2_A321_intdors.suaveVehicle.mass_properties.max_zero_fuel += A321_deltaMZFW
    H2_A321_intdors.ConvertToLH2(tankStyle="Both",
                                dorsalTank=dorsTank2, dorsalxsecNum=10, dorsalxStart=3.0,
                                Sfront=2.5, Dfront=0.15, Saft=3.5, Daft=0.35,
                                internalTank=internalTank)
    """

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
if True:
    Aircrafts = list([A320])#, H2_A320_MZFWplusL)
    colours = ("red", "blue")
    i = 0

    for Aircraft in Aircrafts:
        results = aeroSweep(vehicle = Aircraft.suaveVehicle,
                            alphas = np.linspace(-1*np.pi/180, 6*np.pi/180, 1000),
                            machs = np.array([0.3, 0.6, Aircraft.cruise_mach]),
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

        fig, ax2 = plt.subplots(1, 1, dpi=200)
        #ax2 = fig.add_subplot(121)
        #ax3 = fig.add_subplot(122)
        #fig.tight_layout()
        #fig.subplots_adjust(wspace=0)

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
                     f" $C_D$ = {np.sum(dragBreakdown):.3f}")
        """
        # Component breakdown for parasitic drag
        parasiticDragFull = results["parasiticDragFull"][nearestMachIndex][nearestClIndex]

        parasiticBreakdown = [parasiticDragFull["wings"]["htail"]["wing"][0,0,0],
                              parasiticDragFull["wings"]["vtail"]["wing"][0,0,0],
                              parasiticDragFull["wings"]["main_wing"]["wing"][0,0,0]]
        parasiticLabels = ["Horizontal tail",
                           "Vertical tail",
                           "Wing"]

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

        fig, ax = plt.subplots(1, 1, dpi=200)
        bottom = 0
        for dragType, dragLabel in zip(dragBreakdown, dragLabels):
            p = ax.bar("Total", dragType, width=1, bottom=bottom)
            bottom += dragType

            ax.bar_label(p, labels=dragLabel, label_type="center")
        """


    plt.show()

# Fixed range cruise flight; given payload, determine required fuel
if True:
    results = fixedRangeMission(Aircraft = A320,
                                range = 500 * Units.km,
                                payload = 5000 * Units.kg,
                                ICAOreserves = True,
                                climbType = "AircraftDefined",
                                cruiseAlt = 10667 * Units.m)
    
    plot_mission(results)
    plt.show()
    
# Produce payload range diagrams
if False:
    ncols = 80
    Aircrafts = (A320, H2_A320_MZFWplusS, H2_A320_MZFWplusL)
    colours = ("red", "lightblue", "navy")
    i = 0

    fig, ax = plt.subplots(1, 1, dpi=200)

    pbar1 = tqdm.tqdm(total=len(Aircrafts), position=1, desc=f"Payload-range diagram", ncols=ncols)
    pbars = []
    for Aircraft in Aircrafts:
        vehicle = Aircraft.suaveVehicle

        payloads = [vehicle.mass_properties.max_payload, vehicle.mass_properties.max_payload, 0, 0]
        fuels = [0, 0, vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]
        
        payloads[2] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + fuels[2])
        fuels[1] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + payloads[1])

        for k in range(len(payloads)):
            payload = payloads[k]
            if payload < 0:
                print(f"Invalid payload {payload/1000:.2f} tonnes, setting to zero.")
                payloads[k] = 0

        ranges = np.zeros_like(payloads)

        pbars.append(tqdm.tqdm(total=len(payloads)-1, position=2+i, desc=f"  {Aircrafts[i].dispName}", ncols=ncols))
        for j in range(1, len(payloads)):
            if fuels[j] == 0:
                ranges[j] = 0
            else:
                t_max = endurance(Aircraft, payload=payloads[j], fuel=fuels[j])

                # Reserves for ICAO Annex 6 without a destination alternate aerodrome
                fuelReserve = np.max((fuels[j]*5*60/t_max, 0.05*fuels[j])) + ((30+15)*60)*fuels[j]/t_max

                results =  fixedFuelMission(Aircraft = Aircraft,
                                            fuel = fuels[j],
                                            fuelReserve = fuelReserve,
                                            payload = payloads[j],
                                            climbType = "AircraftDefined",
                                            cruiseAlt = 10667 * Units.m)

                ranges[j] = results.segments[-1].conditions.frames.inertial.aircraft_range[-1,0]
            
            pbars[-1].update(1)

        ax.plot(np.divide(ranges, 1000), np.divide(payloads, 1000), color=colours[i], label=Aircraft.dispName)
        i += 1
        pbar1.update(1)
    tqdm.tqdm.write("Finished building payload-range diagram.")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")
    ax.legend(("A320ceo, kerosene", "A320 deriv. 7.6T LH2 tank", "A320 deriv. 4.0T LH2 tank"))
    ax.grid()

    plt.show()

# FPPR spectrum
# Fly along the payload-range spectrum
if False:
    fig, ax = plt.subplots(1, 1, dpi=200)

    nflights = 20
    Aircrafts = (A320, H2_A320_MZFWplusS, H2_A320_MZFWplusL) #A320

    for Aircraft in Aircrafts:
        # Min reserve fuel to be compliant
        residual = 1
        i = 0
        while residual > 0.01:
            if i == 0:
                fuelGuess = Aircraft.suaveVehicle.mass_properties.max_fuel/10
            min_endurance = endurance(Aircraft, payload=Aircraft.suaveVehicle.mass_properties.max_payload, fuel=fuelGuess)
            fuelReserve = np.max((fuelGuess*5*60/min_endurance, 0.05*fuelGuess)) + ((30+15)*60)*fuelGuess/min_endurance

            residual = np.abs(1 - fuelGuess/fuelReserve)
            fuelGuess *= fuelReserve/fuelGuess
            i += 1

        fuels = np.linspace(fuelGuess*1.1, Aircraft.suaveVehicle.mass_properties.max_fuel, nflights)
        fuelBurns = np.zeros_like(fuels)
        ranges = np.zeros_like(fuels)

        payloads = []
        for fuel in fuels:
            #payloads.append(11500)
            payloads.append(np.min([Aircraft.suaveVehicle.mass_properties.max_takeoff - fuel - Aircraft.suaveVehicle.mass_properties.operating_empty,
                                    Aircraft.suaveVehicle.mass_properties.max_payload]))

        for i in range(nflights):

            t_max = endurance(Aircraft, payload=payloads[i], fuel=fuels[i])
            # Reserves for ICAO Annex 6 without a destination alternate aerodrome
            fuelReserve = np.max((fuels[i]*5*60/t_max, 0.05*fuels[i])) + ((30+15)*60)*fuels[i]/t_max

            results = fixedFuelMission(Aircraft = Aircraft,
                                       fuel = fuels[i],
                                       fuelReserve = fuelReserve,
                                       payload = payloads[i],
                                       climbType = "AircraftDefined",
                                       cruiseAlt = 10667 * Units.m)

            ranges[i] = results.segments[-1].conditions.frames.inertial.aircraft_range[-1,0]
            fuelBurns[i] = results.segments[0].conditions.weights.total_mass[0,0] - results.segments[-1].conditions.weights.total_mass[-1,0]
            #print(ranges)
            
        FEPPRs = np.divide(np.multiply(fuelBurns, Aircraft.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy), np.multiply(ranges, payloads))
        ax.plot(np.divide(ranges, 1000), FEPPRs, label=Aircraft.dispName)

    ax.grid()
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Fuel energy, J/kgm")
    ax.set_ylim(0)
    ax.set_xlim(0)
    ax.legend()
    plt.show()