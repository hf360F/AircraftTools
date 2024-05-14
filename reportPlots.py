import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

import pickle
import baseline as base
import derivative as dv
from flightFunctions import *
import SUAVE
import openvsp as vsp
from SUAVE.Core import Data, Units
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
from copy import deepcopy


## Baseline aircraft definitions
A320 = base.Baseline("Airbus A320-200ceo")

## Style
colourA320kero = "red"
fsizefull = 16
ddpi = 200

## Global flight distance distribution
if False:
    binsnm = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 6000, 7000, 8000]
    binskm = np.multiply(binsnm, 1.852)

    # Centre of each bin
    binskmcentres = np.zeros_like(binskm[:-1])
    for i in range(len(binskmcentres)):
        binskmcentres[i] = (binskm[i] + binskm[i+1])/2

    scale = 607/14 # points per million pax per week
    points = [327, 524, 329, 215, 107, 60, 48, 33, 30, 9, 20, 42, 28, 28, 6, 2]
    pax = np.divide(points, scale) 

    # Total weekly pax km in each bin
    binestseatkm = np.multiply(binskmcentres, pax)

    # Cumulative values
    cumxs = np.insert(binskmcentres, 0, 0)
    cumseatkm = np.insert(np.cumsum(binestseatkm), 0, 0)
    cumseatkm = np.divide(cumseatkm, np.max(cumseatkm))

    fig, (ax, ax2) = plt.subplots(2, 1, dpi=ddpi, sharex="col")
    ax3 = ax2.twinx()

    ax.hist(binskm[:-1], binskm, weights=pax, color="blue", zorder=100)

    l1 = ax2.plot(cumxs, cumseatkm, color="red", label="Cumulative")    
    l2 = ax3.plot(binskmcentres, np.divide(binestseatkm, np.sum(binestseatkm)), color="green", label="Density")
    
    ax.set_ylabel("Seats per week, millions")
    ax2.set_xlabel("Flight distance, km")
    ax2.set_ylabel("Cum. fraction of all seat-km")
    ax3.set_ylabel("Relative seat-km density")

    ax.set_xlim(np.min(binskm), np.max(binskm))
    ax.set_ylim(0)
    ax2.set_ylim(0)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0)
    ax.grid()
    ax2.grid()

    lns = l1+l2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)

    plt.show()

## A320 custom payload range
if False:
    fig, ax = plt.subplots(1, 1, dpi=ddpi)

    Aircraft = A320
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
    reserves = np.zeros_like(payloads)

    reserves[0] = ICAOreserve(Aircraft, payloads[0], 1E-5)

    for j in range(1, len(payloads)):
        if fuels[j] == 0:
            ranges[j] = 0
            reserves[j] = ICAOreserve(Aircraft, payloads[j], 0)
        else:
            reserves[j] = ICAOreserve(Aircraft, payloads[j], fuels[j])

            results =  fixedFuelMission(Aircraft = Aircraft,
                                        fuel = fuels[j],
                                        fuelReserve = reserves[j],
                                        payload = payloads[j],
                                        climbType = "AircraftDefined",
                                        cruiseAlt = 10667 * Units.m)

            ranges[j] = results.segments[-1].conditions.frames.inertial.aircraft_range[-1,0]
    
    alphaFill = 0.6
    fsize = fsizefull

    missionFuels = np.subtract(fuels, reserves)
    for i in range(len(missionFuels)):
        if missionFuels[i] < 0:
            missionFuels[i] = 0

    OEW = (Aircraft.suaveVehicle.mass_properties.operating_empty/1000)
    
    ax.fill_between((0, np.max(ranges)/1000), (0, 0), (OEW, OEW), color="green", alpha=alphaFill)
    ax.plot((0, np.max(ranges)/1000), (OEW, OEW), color="green")
    ax.text(np.max(ranges)/2000, OEW/2, "Operating Empty Weight", verticalalignment="center", horizontalalignment="center", fontsize = fsize)

    ax.plot(np.divide(ranges, 1000), np.add(np.divide(payloads,1000), OEW), color="blue")
    ax.fill_between(np.divide(ranges, 1000), (OEW, OEW, OEW, OEW), np.add(np.divide(payloads,1000), OEW), color="blue", alpha=alphaFill)
    ax.text(np.max(ranges)/2000, OEW + np.max(payloads)/2000, "Payload", verticalalignment="center", horizontalalignment="center", fontsize = fsize)

    ax.plot(np.divide(ranges, 1000), np.add(np.divide(reserves,1000), np.add(OEW, np.divide(payloads,1000))), color="yellow")
    ax.fill_between(np.divide(ranges, 1000), np.add(np.divide(payloads,1000), OEW), np.add(np.divide(reserves,1000), np.add(OEW, np.divide(payloads,1000))), color="yellow", alpha=alphaFill)
    ax.text(0.75*np.max(ranges)/1000, (OEW + np.max(payloads)/1000 + np.max(reserves)/2000)*0.93, "ICAO Fuel Reserve", verticalalignment="center", horizontalalignment="center", fontsize = fsize/2, rotation = -10)

    ax.plot(np.divide(ranges, 1000), np.add(np.divide(missionFuels, 1000), np.add(np.divide(reserves,1000), np.add(OEW, np.divide(payloads,1000)))), color="red")
    ax.fill_between(np.divide(ranges, 1000), np.add(np.add(np.divide(payloads,1000), OEW), np.divide(reserves, 1000)), np.add(np.divide(missionFuels, 1000), np.add(np.divide(reserves,1000), np.add(OEW, np.divide(payloads,1000)))), color="red", alpha=alphaFill)
    ax.text(0.65*np.max(ranges)/1000, OEW + np.max(payloads)/1000 + 0.96*np.max(missionFuels)/2000 - np.max(reserves)/1000, "Mission Fuel", verticalalignment="center", horizontalalignment="center", fontsize = fsize)

    ax.axhline(Aircraft.suaveVehicle.mass_properties.max_takeoff/1000, linestyle="dashed", color="black")
    ax.text(0.98*np.max(ranges)/1000, Aircraft.suaveVehicle.mass_properties.max_takeoff/1000, "Max takeoff", fontsize = fsize/2, verticalalignment="bottom", horizontalalignment="right")
    ax.axhline(Aircraft.suaveVehicle.mass_properties.max_zero_fuel/1000, linestyle="dashed", color="black")
    ax.text(0.98*np.max(ranges)/1000, Aircraft.suaveVehicle.mass_properties.max_zero_fuel/1000, "Max zero fuel", fontsize = fsize/2, verticalalignment="bottom", horizontalalignment="right")

    # Add the design point
    ax.scatter(np.divide(ranges[1:3], 1000), np.add(OEW, np.divide(payloads[1:3], 1000)), s=50, marker="o", color="black", edgecolors="white", linewidth=2, zorder=100)
    ax.annotate("A", (ranges[1]/1000, OEW+payloads[1]/1000),zorder=200)
    ax.annotate("B", (ranges[2]/1000, OEW+payloads[2]/1000),zorder=200)

    #ax.plot(np.divide(ranges, 1000), np.divide(payloads, 1000), color="black", label="Payload")
    #ax.plot(np.divide(ranges, 1000), np.divide(fuels, 1000), color="red", label="Fuel")
    #ax.plot(np.divide(ranges, 1000), np.)

    ax.set_xlim(0, np.max(ranges)/1000)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Mass, tonnes")
    #ax.set_title(f"Payload-range diagram for {Aircraft.dispName}")
    #ax.legend()
    ax.grid()

    plt.show()

## A320 FEPR contours
if True:
    generateNew = True
    fname = "FEPRdata.pickle"

    if generateNew:
        # Check if we want to save
        answer = input(f"Pickle data: Y or N? (will overwrite existing {fname})")
        if answer == "Y" or answer == "y":
            saveFlag = True
        elif answer == "N" or answer == "n": 
            saveFlag = False
        else: tqdm.tqdm.write("Please enter Y or N.")

        Aircraft = A320
        vehicle = Aircraft.suaveVehicle

        rangeCounts = 10
        payloadCounts = 6
        points = rangeCounts*payloadCounts

        pbar1 = tqdm.tqdm(total=points, position=1, desc=f"Contour points", ncols=80)

        frontierPayloads = [vehicle.mass_properties.max_payload, vehicle.mass_properties.max_payload, 0, 0]
        frontierFuels = [0, 0, vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]
            
        frontierPayloads[2] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + frontierFuels[2])
        frontierFuels[1] = vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + frontierPayloads[1])
        frontierRanges = np.zeros_like(frontierFuels)

        frontierReserves = np.zeros_like(frontierFuels)
        frontierReserves[0] = ICAOreserve(Aircraft, frontierPayloads[0], 1E-5)

        # Obtain frontier points 
        for j in range(1, len(frontierPayloads)):
            if frontierFuels[j] == 0:
                frontierRanges[j] = 0
                frontierReserves[j] = ICAOreserve(Aircraft, frontierPayloads[j], 0)
            else:
                frontierReserves[j] = ICAOreserve(Aircraft, frontierPayloads[j], frontierFuels[j])

                results =  fixedFuelMission(Aircraft = Aircraft,
                                            fuel = frontierFuels[j],
                                            fuelReserve = frontierReserves[j],
                                            payload = frontierPayloads[j],
                                            climbType = "AircraftDefined",
                                            cruiseAlt = 10667 * Units.m)

                frontierRanges[j] = results.segments[-1].conditions.frames.inertial.aircraft_range[-1,0]

        contRanges = np.multiply(np.linspace(0.5/rangeCounts, 1 - 0.5/rangeCounts, rangeCounts), np.max(frontierRanges))
        contPayloadRangeFEPR = np.zeros((rangeCounts, payloadCounts, 3))

        for i in np.flip(range(rangeCounts)):
            flightRange = contRanges[i]
            
            for j in np.flip(range(payloadCounts)):
                if flightRange <= frontierRanges[1]:
                    maxPayload = frontierPayloads[1]
                elif flightRange <= frontierRanges[2]:
                    maxPayload = frontierPayloads[1] - (flightRange - frontierRanges[1])*(frontierPayloads[1] - frontierPayloads[2])/(frontierRanges[2] - frontierRanges[1])
                elif flightRange <= frontierRanges[3]:
                    maxPayload = (frontierRanges[3] - flightRange)*frontierPayloads[2]/(frontierRanges[3]-frontierRanges[2])
                else:
                    raise ValueError(f"Range {flightRange/1000:.1f} km exceeds ferry range {frontierRanges/1000:.1f} km.")

                contPayloadRangeFEPR[i,j,0] = maxPayload*(j+0.5)/(payloadCounts-0.5)
                contPayloadRangeFEPR[i,j,1] = flightRange

                if contPayloadRangeFEPR[i,j,0] == 0 or contPayloadRangeFEPR[i,j,1] == 0:
                    contPayloadRangeFEPR[i,j,2] == np.nan
                else:
                    results = fixedRangeMission(Aircraft = Aircraft,
                                                range = contPayloadRangeFEPR[i,j,1],
                                                payload = contPayloadRangeFEPR[i,j,0],
                                                ICAOreserves = True,
                                                climbType = "AircraftDefined",
                                                cruiseAlt = 10667 * Units.m)

                    initialMass = results.segments[0].conditions.weights.total_mass[0,0]
                    finalMass = results.segments[-1].conditions.weights.total_mass[-1,0]
                    fuelBurn = initialMass - finalMass

                    contPayloadRangeFEPR[i,j,2] = fuelBurn*Aircraft.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy/(contPayloadRangeFEPR[i,j,0]*contPayloadRangeFEPR[i,j,1])

                if saveFlag:
                    data = {"frontierRanges": frontierRanges,
                    "frontierPayloads": frontierPayloads,
                    "contPayloadRangeFEPR": contPayloadRangeFEPR}

                    with open(fname, "wb") as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

                pbar1.update(1)

        # Check if we want to save
        answer = input(f"Pickle data: Y or N? (will overwrite existing {fname})")
        if answer == "Y" or answer == "y":

            data = {"frontierRanges": frontierRanges,
            "frontierPayloads": frontierPayloads,
            "contPayloadRangeFEPR": contPayloadRangeFEPR}

            with open(fname, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        elif answer == "N" or answer == "n": 
            pass
        else: tqdm.tqdm.write("Please enter Y or N.")        
        
    else:
        with open(fname, "rb") as f:
            data = pickle.load(f)

        # Unpack    
        frontierRanges = data["frontierRanges"]
        frontierPayloads = data["frontierPayloads"]
        contPayloadRangeFEPR = data["contPayloadRangeFEPR"]

    minFEPR = np.min(contPayloadRangeFEPR[:,:,2])
    # Should evaluate min FEPR by checking FEPR at R1 and R2 with respective max payloads

    normalisedFEPR = (contPayloadRangeFEPR[:,:,2]/minFEPR - 1)*100

    fig, ax = plt.subplots(1, 1, dpi=ddpi)
    ax.plot(np.divide(frontierRanges, 1000), np.divide(frontierPayloads, 1000), color="blue")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")

    ax.grid()

    ax.scatter(np.divide(contPayloadRangeFEPR[:,:,1], 1000), np.divide(contPayloadRangeFEPR[:,:,0], 1000), marker="x", color="black")
    CS = ax.contour(np.divide(contPayloadRangeFEPR[:,:,1], 1000), 
                    np.divide(contPayloadRangeFEPR[:,:,0], 1000),
                    normalisedFEPR,
                    levels=np.multiply(np.subtract((1.05, 1.10, 1.25, 1.50, 2.00, 3.00, 6.00), 1), 100),
                    colors="black")
    
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"+{s}\%" if plt.rcParams["text.usetex"] else f"+{s}%"

    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, manual=True)

    plt.show()