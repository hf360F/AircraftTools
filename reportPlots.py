import numpy as np
import scipy.interpolate
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
colours = ("blue", "green", "firebrick")
fsizefull = 16
ddpi = 200

## Global flight distance distribution
if False:
    binsnm = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 6000, 7000, 8000]
    binskm = np.multiply(binsnm, 1.852)
    binwidthskm = np.subtract(binskm[1:], binskm[:-1])

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
    l2 = ax3.plot(binskmcentres, np.divide(np.divide(binestseatkm, binwidthskm), np.sum(binestseatkm)), color="green", label="Density")
    
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

## Distributions
if True:
    show = False
    # Air service one, Sector length distribution comparison for Ryanair, easyJet and Wizz Air; similarities and differences
    rangeBoundsRyanair = [142, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3250, 3500, 4093]
    rangeBoundsEasyjet = [122, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3250, 3500, 4147]
    
    rangesRyanair = np.zeros(len(rangeBoundsRyanair)-1)
    rangesEasyjet = np.zeros_like(rangesRyanair)

    colourRyanair = np.divide((7, 53, 144), 255)
    colourEasyjet = np.divide((255, 102, 0), 255)

    for i in range(len(rangeBoundsRyanair)-1):
        rangesRyanair[i] = (rangeBoundsRyanair[i+1] + rangeBoundsRyanair[i])/2
        rangesEasyjet[i] = (rangeBoundsEasyjet[i+1] + rangeBoundsEasyjet[i])/2

    freqsRyanair = np.divide([2.922, 7.394, 9.374, 11.08, 12.48, 11.52, 10.53, 10.77, 6.985, 4.356, 3.093, 2.410, 2.103, 2.205, 1.625, 0.4301, 0.4301], 100)
    freqsRyanair = np.divide(freqsRyanair, np.sum(freqsRyanair)) # Rescale to sum to 100%
    wfreqsRyanair = np.multiply(freqsRyanair, rangesRyanair)
    wfreqsRyanair = np.divide(wfreqsRyanair, np.sum(wfreqsRyanair))

    freqsEasyjet = np.divide([4.083, 10.19, 9.647, 11.12, 9.579, 9.067, 8.418, 7.087, 5.414, 3.878, 3.332, 2.854, 3.673, 4.868, 3.025, 2.048, 1.796], 100)
    freqsEasyjet = np.divide(freqsEasyjet, np.sum(freqsEasyjet)) # Rescale to sum to 100%
    wfreqsEasyjet = np.multiply(freqsEasyjet, rangesEasyjet)
    wfreqsEasyjet = np.divide(wfreqsEasyjet, np.sum(wfreqsEasyjet))

    if show:
        fig, (ax, ax3) = plt.subplots(2, 1, dpi=ddpi, sharex=True)
        #ax2 = ax.twinx()
        #ax4 = ax3.twinx()

        l1 = ax.plot(rangesRyanair, freqsRyanair, color=colourRyanair, label="Ryanair")
        #l2 = ax2.plot(rangesRyanair, np.cumsum(freqsRyanair), color="blue", linestyle="dashed", label="Ryanair cum.")
        l3 = ax.plot(rangesEasyjet, freqsEasyjet, color=colourEasyjet, label="easyJet")
        #l4 = ax2.plot(rangesEasyjet, np.cumsum(freqsEasyjet), color="orange", linestyle="dashed", label="easyJet cum.")

        #ax.set_xlim(0)
        ax.set_ylim(0)
        #ax2.set_ylim(0)
        #ax.set_xlabel("Range, km")
        ax.set_ylabel("Relative flight density")
        #ax2.set_ylabel("Cum. fraction of flights")
        ax.grid()

        lns = l1+l3#l1+l2+l3+l4
        labs = []
        for l in lns:
            labs.append(l.get_label())
        ax.legend(lns, labs, loc="best") #loc="center right")

        l5 = ax3.plot(rangesRyanair, wfreqsRyanair, color=colourRyanair)
        l6 = ax3.plot(rangesEasyjet, wfreqsEasyjet, color=colourEasyjet)
        #l7 = ax4.plot(rangesRyanair, np.cumsum(wfreqsRyanair), color="blue", linestyle="dashed")
        #l8 = ax4.plot(rangesEasyjet, np.cumsum(wfreqsEasyjet), color="orange", linestyle="dashed")

        ax3.set_xlim(0)
        ax3.set_ylim(0)
        ax3.set_xlabel("Range, km")
        ax3.set_ylabel("Relative flight-km density")
        #ax4.set_ylabel("Cum. fraction of km flown")
        ax3.grid()

        plt.show()

## A320 custom weight buildup diagram
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

## H320 payload range and multi-tank sizing
if True:
    # to add:
    # constant payload line
    # FEPR across constant payload line and along max payload frontier
    generateNew = True
    fname = "multiLH2_10tanks.pickle"

    if generateNew:
        constPayload = 180*84 # 180 passengers by 84 kg / passenger
        constPayloadPoints = 20 # Number of ranges at which to fly along const payload line, per aircraft

        etaGravMax = 0.60 # for full size tank only
        deltaMZFW = 11300 # A321 uprating from A320

        fuelMasses = [7400, 6575, 5750, 4925, 4100, 3275, 2450, 1625, 800] # 7400/5000/3000 ?# kg, must be descending order (flipped later)
        #fuelMasses = [2000, 1000]
        Aircrafts = []
        i = 0

        maxTankLength = 17 # metres
        designTankDia = 3.2 # m
        
        AR = maxTankLength/designTankDia
        diaTol = 1E-3

        # Check if we want to save
        answer = input(f"Pickle data (will overwrite any existing {fname}): Y or N? ")
        if answer == "Y" or answer == "y":
            saveFlag = True
        elif answer == "N" or answer == "n": 
            saveFlag = False
        else: tqdm.tqdm.write("Please enter Y or N.")

        # Generate tanks and aircraft
        for fuelMass in fuelMasses:
            if i == 0:
                etaGrav = etaGravMax
            else:                
                cylLH2frac = (fuelMass - endUsableLH2)/fuelMass # Note we are assuming same effective LH2 density
                endsLH2frac = 1 - cylLH2frac

                etaGrav = 1/(1 + (1/cylEtaGrav - 1)*cylLH2frac + (1/endsEtaGrav - 1)*endsLH2frac)
            
            deriv = dv.Derivative(A320, f"H2_{fuelMass}kg")
            tankDia = 0

            while tankDia/designTankDia < 1-diaTol or tankDia/designTankDia > 1+diaTol:
                tank = dv.Tank(usableLH2=fuelMass,
                               ventPressure=1.5,
                               aspectRatio=AR,
                               ullageFraction=0.05,
                               endGeometry="2:1elliptical",
                               fidelity="AutoInsulation",
                               etaGrav=etaGrav,
                               mdot_boiloff=0.03611,
                               t_wall=0.005,
                               show=False,
                               verbose=False)
                tankDia = tank.Do
                AR = tank.aspectRatio / (designTankDia/tankDia)

            if tank.Lo > maxTankLength:
                raise ValueError(f"Tank length {tank.Lo:.1f} m exceeds maximum {maxTankLength:.1f} m")

            if i == 0:
                maxCapacity = tank.tankCapacity

                endUsableLH2 = (2*tank.k_end*(tank.Di/1)**3/tank.tankCapacity)*tank.usableLH2
                endStruct = tank.Aends*tank.m_struct/tank.Awet # Same wall thickness assumed
                endsEtaGrav = endUsableLH2/(endUsableLH2+endStruct)

                cylUsableLH2 = tank.usableLH2 - endUsableLH2
                cylStruct = tank.m_struct - endStruct
                cylEtaGrav = cylUsableLH2/(cylUsableLH2+cylStruct)
            
            i += 1
            tqdm.tqdm.write(f"Tank for {fuelMass:.1f} kg, {tank.Do:.2f} m x {tank.Lo:.2f} m, gravimetric efficiency {tank.etaGrav:.3f}")

            deriv.ConvertToLH2(tankStyle="DorsalOnly",
                               dorsalTank=tank, dorsalxsecNum=10, dorsalxStart=3.0,
                               Sfront=2.5, Dfront=0.15, Saft=3.5, Daft=0.35)
            
            deriv.suaveVehicle.mass_properties.max_zero_fuel += deltaMZFW
            deriv.suaveVehicle.mass_properties.max_takeoff += deltaMZFW
            deriv.suaveVehicle.mass_properties.max_payload = deriv.suaveVehicle.mass_properties.max_zero_fuel - deriv.suaveVehicle.mass_properties.operating_empty

            Aircrafts.append(deepcopy(deriv))

        # Easier to work in ascending order generally
        Aircrafts = np.flip(Aircrafts)
        fuelMasses = np.flip(fuelMasses)

        pbar1 = tqdm.tqdm(total=len(Aircrafts), position=1, desc=f"Payload range", ncols=80)

        aircraftDatas = []

        for Aircraft in Aircrafts:
            vehicle = Aircraft.suaveVehicle

            # Setup payload range frontier masses based on MZFW = MTOW constraint
            frontierFuels = [0, vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]

            frontierReserves = np.zeros_like(frontierFuels)
            frontierReserves[0] = ICAOreserve(Aircraft, vehicle.mass_properties.max_payload, 1E-5)
            frontierReserves[1] = ICAOreserve(Aircraft, vehicle.mass_properties.max_payload-vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel)
            frontierReserves[2] = ICAOreserve(Aircraft, 0, vehicle.mass_properties.max_fuel)

            frontierPayloads = np.zeros_like(frontierFuels)
            frontierPayloads[0] = vehicle.mass_properties.max_payload - frontierFuels[0] - frontierReserves[0]
            frontierPayloads[1] = vehicle.mass_properties.max_payload - frontierFuels[1] - frontierReserves[1]
            frontierPayloads[2] = 0

            frontierRanges = np.zeros_like(frontierFuels)

            # Obtain frontier points 
            for j in range(1, len(frontierPayloads)):
                if frontierFuels[j] == 0:
                    frontierRanges[j] = 0
                else:
                    results =  fixedFuelMission(Aircraft = Aircraft,
                                                fuel = frontierFuels[j],
                                                fuelReserve = frontierReserves[j],
                                                payload = frontierPayloads[j],
                                                climbType = "AircraftDefined",
                                                cruiseAlt = 10667 * Units.m)

                    frontierRanges[j] = results.segments[-1].conditions.frames.inertial.aircraft_range[-1,0]

            fuelLCV = Aircraft.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy

            aircraftData = {"frontierFuels": frontierFuels,
                            "frontierReserves": frontierReserves,
                            "frontierRanges": frontierRanges,
                            "frontierPayloads": frontierPayloads,
                            "fuelLCV": fuelLCV}
            
            aircraftDatas.append(aircraftData)
            pbar1.update(1)      

        def maxRange(payload, frontierRanges, frontierPayloads):
            if payload > frontierPayloads[0]:
                raise ValueError(f"Payload {payload:.1f} kg exceeds max payload {frontierPayloads[0]:.1f} kg.")
            elif payload >= frontierPayloads[1]:
                return (frontierPayloads[0] - payload)/(frontierPayloads[0] - frontierPayloads[1])*frontierRanges[1]
            elif payload >= 0:
                return frontierRanges[1] + (frontierPayloads[1] - payload)/frontierPayloads[1]*(frontierRanges[2] - frontierRanges[1])
            else:
                raise ValueError(f"Payload {payload:.1f} kg is not positive.")

        maxRangesConstPayload = []
        for i in range(len(fuelMasses)):
            maxRangesConstPayload.append(maxRange(constPayload, aircraftDatas[i]["frontierRanges"], aircraftDatas[i]["frontierPayloads"]))

        pbar2 = tqdm.tqdm(total=len(fuelMasses)*constPayloadPoints, position=1, desc=f"Constant payload points", ncols=80)

        constPayloadRangesFuels = np.zeros((len(fuelMasses), constPayloadPoints, 2))
        
        for i in range(len(fuelMasses)):
            aircraftRanges = np.linspace(20E3, maxRangesConstPayload[i], constPayloadPoints)
            
            for j in range(len(aircraftRanges)):
                constPayloadRangesFuels[i, j, 0] = aircraftRanges[j]
                results = fixedRangeMission(Aircraft = Aircrafts[i],
                                            range = constPayloadRangesFuels[i,j,0],
                                            payload = constPayload,
                                            ICAOreserves = True,
                                            climbType = "AircraftDefined",
                                            cruiseAlt = 10667 * Units.m)

                initialMass = results.segments[0].conditions.weights.total_mass[0,0]
                finalMass = results.segments[-1].conditions.weights.total_mass[-1,0]
                fuelBurn = initialMass - finalMass
                constPayloadRangesFuels[i,j,1] = fuelBurn

                pbar2.update(1)

        if saveFlag:
            data = {"aircraftDatas": aircraftDatas,
                    "fuelMasses": fuelMasses,
                    "constPayload": constPayload,
                    "maxRangesConstPayload": maxRangesConstPayload,
                    "constPayloadRangesFuels": constPayloadRangesFuels}

            with open(fname, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # Load data
    with open(fname, "rb") as f:
        data = pickle.load(f)

        # Unpack
        aircraftDatas = data["aircraftDatas"]
        fuelMasses = data["fuelMasses"]
        constPayload = data["constPayload"]
        constPayloadRangesFuels = data["constPayloadRangesFuels"]
        maxRangesConstPayload = data["maxRangesConstPayload"]

    fig, (ax, ax2) = plt.subplots(2, 1, dpi=ddpi, sharex=True)

    i = 0
    for aircraftData in aircraftDatas:
        frontierRanges = aircraftData["frontierRanges"]
        frontierPayloads = aircraftData["frontierPayloads"]
        if i != 0:
            # Find intersection of payload range frontiers
            dataPrev = aircraftDatas[i-1]
            rangesPrev = dataPrev["frontierRanges"]
            payloadsPrev = dataPrev["frontierPayloads"]

            endSlopePrev = (payloadsPrev[-1] - payloadsPrev[-2])/(rangesPrev[-1] - rangesPrev[-2])
            endOffsetPrev = -endSlopePrev*rangesPrev[-1]
            frontSlopeNew = (frontierPayloads[1] - frontierPayloads[0])/(frontierRanges[1] - frontierRanges[0])
            frontOffsetNew = frontierPayloads[0]

            intersectRange = (frontOffsetNew - endOffsetPrev)/(endSlopePrev - frontSlopeNew)
            intersectPayload = frontOffsetNew + intersectRange*frontSlopeNew

            frontierRanges[0] = intersectRange
            frontierPayloads[0] = intersectPayload

        ax.plot(np.divide(frontierRanges, 1000), np.divide(frontierPayloads, 1000), color=colours[i], label=r"LH$_2$ capacity " + f"{fuelMasses[i]/1000:.1f} T")
        ax.plot()
        i += 1

    # Kerosene reference
    ax.plot((0, 2860, 4650, 5370), (18.325, 18.325, 11.958, 0), color="grey", label="Kerosene")

    # Constant payload line
    ax.axhline(constPayload/1000, color="black", linestyle="dashed")
    ax.text(ax.get_xlim()[1], constPayload*1.02/1000, "Max passengers", fontsize = fsizefull/2, verticalalignment="bottom", horizontalalignment="right")

    for i in range(len(fuelMasses)):
        ax.vlines(maxRangesConstPayload[i]/1000, 0, constPayload/1000, color=colours[i], linestyle="dotted")
        ax2.axvline(maxRangesConstPayload[i]/1000, color=colours[i], linestyle="dotted")

    FEPRinterpolators = []
    xsList = []
    ysList = []

    for i in range(len(fuelMasses)):
        ranges = constPayloadRangesFuels[i,:,0]
        fuels = np.ravel(constPayloadRangesFuels[i,:,1])
        FEPRs = np.divide(np.multiply(fuels, aircraftDatas[0]["fuelLCV"]), np.multiply(ranges, constPayload))

        FEPRinterpolator = scipy.interpolate.interp1d(ranges, FEPRs, kind="quadratic")
        FEPRinterpolators.append(FEPRinterpolator)
        nx = 50

        # for i > 0: want to plot two lines: dashed, up to maxRange(i-1)
        if i > 0:
            xs1 = np.linspace(ranges[0], maxRangesConstPayload[i-1], nx)
            xs2 = np.linspace(maxRangesConstPayload[i-1], ranges[-1], nx)
            xsList.append(xs1)
            xsList.append(xs2)
            ys1 = [FEPRinterpolator(x) for x in xs1]
            ys2 = [FEPRinterpolator(x) for x in xs2]
            ysList.append(ys1)
            ysList.append(ys2)
            
            # Include last point from previous curve
            xs2 = np.concatenate(([xs2[0]], xs2))
            ys2 = np.concatenate(([constPayloadRangesFuels[i-1,-1,1]*aircraftDatas[0]["fuelLCV"]/(constPayloadRangesFuels[i-1,-1,0]*constPayload)], ys2))

            if i == 1:
                xsPrev = xsList[-3]
                ysPrev = ysList[-3]
            else:
                xsPrev = np.concatenate([xsList[-4], xsList[-3]])
                ysPrev = np.concatenate([ysList[-4], ysList[-3]])

            xfill = np.sort(np.concatenate([xsPrev, xs1]))
            yfill1 = np.interp(xfill, xsPrev, ysPrev)
            yfill2 = np.interp(xfill, xs1, ys1)
            ax2.fill_between(np.divide(xfill,1000), yfill1, yfill2, where=yfill1 < yfill2, interpolate=True, color=colours[i-1], alpha=0.3, zorder=100)
                    
            ax2.plot(np.divide(xs1, 1000), np.divide(ys1, 1), color=colours[i], alpha=1, linestyle="dashed")
            ax2.plot(np.divide(xs2, 1000), np.divide(ys2, 1), color="black", alpha=1)

        else:
            xs = np.linspace(ranges[0], ranges[-1], 2*nx)
            xsList.append(xs)
            ys = [FEPRinterpolator(x) for x in xs]
            ysList.append(ys)
            ax2.plot(np.divide(xs, 1000), np.divide(ys, 1), color="black", alpha=1)
            ax2.set_ylim(np.min(ys)*0.98)

    ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[0]*1.20)
    ax2.set_ylabel("FEPR, J/kgm")
    #ax2.set_ylabel(r"FEPR/FEPR$_{min}$")
    ax2.grid()

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax2.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")
    ax.legend()

    ax.grid()
    plt.show()


## H320 FEPR contours and payload range
if False:
    generateNew = False
    fname = "FEPRdata60x12_LH2save.pickle"

    if generateNew:
        # A320 dorsal tank derivative - max ferry range
        H2_A320_stockL = dv.Derivative(A320, "H2_MZFW+0_lrg")

        tankL = dv.Tank(usableLH2=7400.0,
                            ventPressure=1.5,
                            aspectRatio=6.0,
                            ullageFraction=0.05,
                            endGeometry="2:1elliptical",
                            fidelity="AutoInsulation",
                            etaGrav=0.60,
                            mdot_boiloff=0.03611,
                            t_wall=0.005,
                            show=False)
        
        H2_A320_stockL.ConvertToLH2(tankStyle="DorsalOnly",
                                    dorsalTank=tankL, dorsalxsecNum=10, dorsalxStart=3.0,
                                    Sfront=2.5, Dfront=0.15, Saft=3.5, Daft=0.35)

        # Check if we want to save
        answer = input(f"Pickle data (will overwrite any existing {fname}): Y or N? ")
        if answer == "Y" or answer == "y":
            saveFlag = True
        elif answer == "N" or answer == "n": 
            saveFlag = False
        else: tqdm.tqdm.write("Please enter Y or N.")

        Aircraft = H2_A320_stockL
        vehicle = Aircraft.suaveVehicle

        rangeCounts = 60
        payloadCounts = 12
        points = rangeCounts*payloadCounts

        pbar1 = tqdm.tqdm(total=points, position=1, desc=f"Contour points", ncols=80)

        # Setup payload range frontier masses based on MZFW = MTOW constraint
        frontierFuels = [0, vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]

        frontierReserves = np.zeros_like(frontierFuels)
        frontierReserves[0] = ICAOreserve(Aircraft, vehicle.mass_properties.max_payload, 1E-5)
        frontierReserves[1] = ICAOreserve(Aircraft, vehicle.mass_properties.max_payload-vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel)
        frontierReserves[2] = ICAOreserve(Aircraft, 0, vehicle.mass_properties.max_fuel)

        frontierPayloads = np.zeros_like(frontierFuels)
        frontierPayloads[0] = vehicle.mass_properties.max_payload - frontierFuels[0] - frontierReserves[0]
        frontierPayloads[1] = vehicle.mass_properties.max_payload - frontierFuels[1] - frontierReserves[1]
        frontierPayloads[2] = 0

        frontierRanges = np.zeros_like(frontierFuels)

        # Obtain frontier points 
        for j in range(1, len(frontierPayloads)):
            if frontierFuels[j] == 0:
                frontierRanges[j] = 0
            else:
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
                    maxPayload = frontierPayloads[0] - (frontierPayloads[0] - frontierPayloads[1])*(flightRange - frontierRanges[0])/(frontierRanges[1] - frontierRanges[0])
                elif flightRange <= frontierRanges[2]:
                    maxPayload = (frontierRanges[2] - flightRange)*frontierPayloads[1]/(frontierRanges[2]-frontierRanges[1])
                else:
                    raise ValueError(f"Range {flightRange/1000:.1f} km exceeds ferry range {frontierRanges/1000:.1f} km.")

                contPayloadRangeFEPR[i,j,0] = 0.99*maxPayload*(j+0.5)/(payloadCounts-0.5)
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

                    fuelLCV = Aircraft.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy

                    contPayloadRangeFEPR[i,j,2] = fuelBurn*fuelLCV/(contPayloadRangeFEPR[i,j,0]*contPayloadRangeFEPR[i,j,1])

                if saveFlag:
                    data = {"frontierFuels": frontierFuels,
                            "frontierReserves": frontierReserves,
                            "frontierRanges": frontierRanges,
                            "frontierPayloads": frontierPayloads,
                            "contPayloadRangeFEPR": contPayloadRangeFEPR,
                            "fuelLCV": fuelLCV}

                    with open(fname, "wb") as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

                pbar1.update(1)      
        
    else:
        with open(fname, "rb") as f:
            data = pickle.load(f)

        # Unpack
        frontierFuels = data["frontierFuels"]
        frontierReserves = data["frontierReserves"]
        frontierRanges = data["frontierRanges"]
        frontierPayloads = data["frontierPayloads"]
        contPayloadRangeFEPR = data["contPayloadRangeFEPR"]
        fuelLCV = data["fuelLCV"]

    minInData = True

    minFEPR = np.min(contPayloadRangeFEPR[:,:,2])
    for i in range(len(frontierFuels)):
        frontierFEPR = fuelLCV*(frontierFuels[i] - frontierReserves[i])/(frontierRanges[i]*frontierPayloads[i])
        print(minFEPR, frontierFEPR)
        if frontierFEPR < minFEPR and frontierRanges[i]*frontierPayloads[i] > 0:
            minInData = True
            minFEPR = frontierFEPR

    for i in range(len(frontierFuels)):
        for j in range(len(frontierPayloads)):
            if contPayloadRangeFEPR[i,j,2] == 0:
                contPayloadRangeFEPR[i,j,2] = np.nan

    normalisedFEPR = (contPayloadRangeFEPR[:,:,2]/minFEPR - 1)*100

    fig, ax = plt.subplots(1, 1, dpi=ddpi)
    ax.plot(np.divide(frontierRanges, 1000), np.divide(frontierPayloads, 1000), color="blue")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")

    ax.grid()

    #ax.scatter(np.divide(contPayloadRangeFEPR[:,:,1], 1000), np.divide(contPayloadRangeFEPR[:,:,0], 1000), marker="x", color="black")
    CS = ax.contour(np.divide(contPayloadRangeFEPR[:,:,1], 1000), 
                    np.divide(contPayloadRangeFEPR[:,:,0], 1000),
                    normalisedFEPR,
                    levels=np.multiply(np.subtract((1.05, 1.10, 1.25, 1.50, 2.00, 3.00, 6.00), 1), 100),
                    colors="black")
    
    if minInData:
        indices = np.argwhere(minFEPR == contPayloadRangeFEPR[:,:,2])[0]
        print(indices)
        minFEPRrange = contPayloadRangeFEPR[indices[0], indices[1], 1]/1000
        minFEPRpayload = contPayloadRangeFEPR[indices[0], indices[1], 0]/1000
    else:
        minFEPRrange = frontierRanges[1]/1000
        minFEPRpayload = frontierPayloads[1]/1000

    ax.scatter(minFEPRrange, minFEPRpayload, s=50, marker="o", color="black", edgecolors="white", linewidth=2, zorder=100)
    ax.annotate(f"Minimum FEPR = {minFEPR:.2f} J/kgm", (minFEPRrange, minFEPRpayload), zorder=200)

    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"+{s}\%" if plt.rcParams["text.usetex"] else f"+{s}%"

    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, manual=True)

    plt.show()

## A320 FEPR contours
if False:
    generateNew = False
    fname = "FEPRdata60x12save.pickle"

    if generateNew:
        # Check if we want to save
        answer = input(f"Pickle data (will overwrite any existing {fname}): Y or N? ")
        if answer == "Y" or answer == "y":
            saveFlag = True
        elif answer == "N" or answer == "n": 
            saveFlag = False
        else: tqdm.tqdm.write("Please enter Y or N.")

        Aircraft = A320
        vehicle = Aircraft.suaveVehicle

        rangeCounts = 60
        payloadCounts = 12
        points = rangeCounts*payloadCounts

        pbar1 = tqdm.tqdm(total=points, position=1, desc=f"Contour points", ncols=80)

        frontierFuels = [0, vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + vehicle.mass_properties.max_payload), vehicle.mass_properties.max_fuel, vehicle.mass_properties.max_fuel]
        frontierPayloads = [vehicle.mass_properties.max_payload, vehicle.mass_properties.max_payload, vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + vehicle.mass_properties.max_fuel), 0]

        frontierReserves = np.zeros_like(frontierFuels)
        frontierReserves[0] = ICAOreserve(Aircraft, frontierPayloads[0], 1E-5)
        frontierReserves[1] = ICAOreserve(Aircraft, frontierPayloads[1], frontierFuels[1])
        frontierReserves[2] = ICAOreserve(Aircraft, frontierPayloads[2], frontierFuels[2])
        frontierReserves[3] = ICAOreserve(Aircraft, frontierPayloads[3], frontierFuels[3])
            
        frontierRanges = np.zeros_like(frontierFuels)

        # Obtain frontier points 
        for j in range(1, len(frontierPayloads)):
            if frontierFuels[j] == 0:
                frontierRanges[j] = 0
            else:
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

                contPayloadRangeFEPR[i,j,0] = 0.99*maxPayload*(j+0.5)/(payloadCounts-0.5)
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

                    fuelLCV = Aircraft.suaveVehicle.networks.turbofan.combustor.fuel_data.specific_energy

                    contPayloadRangeFEPR[i,j,2] = fuelBurn*fuelLCV/(contPayloadRangeFEPR[i,j,0]*contPayloadRangeFEPR[i,j,1])

                if saveFlag:
                    data = {"frontierFuels": frontierFuels,
                            "frontierReserves": frontierReserves,
                            "frontierRanges": frontierRanges,
                            "frontierPayloads": frontierPayloads,
                            "contPayloadRangeFEPR": contPayloadRangeFEPR,
                            "fuelLCV": fuelLCV}

                    with open(fname, "wb") as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

                pbar1.update(1)      
        
    else:
        with open(fname, "rb") as f:
            data = pickle.load(f)

        # Unpack
        frontierFuels = data["frontierFuels"]
        frontierReserves = data["frontierReserves"]
        frontierRanges = data["frontierRanges"]
        frontierPayloads = data["frontierPayloads"]
        contPayloadRangeFEPR = data["contPayloadRangeFEPR"]
        fuelLCV = data["fuelLCV"]

    minFEPR = np.min(contPayloadRangeFEPR[:,:,2])
    for i in range(len(frontierFuels)):
        frontierFEPR = fuelLCV*(frontierFuels[i] - frontierReserves[i])/(frontierRanges[i]*frontierPayloads[i])
        print(minFEPR, frontierFEPR)
        if frontierFEPR < minFEPR and frontierRanges[i]*frontierPayloads[i] > 0:
            minFEPR = frontierFEPR

    for i in range(len(frontierFuels)):
        for j in range(len(frontierPayloads)):
            if contPayloadRangeFEPR[i,j,2] == 0:
                contPayloadRangeFEPR[i,j,2] = np.nan

    normalisedFEPR = (contPayloadRangeFEPR[:,:,2]/minFEPR - 1)*100

    fig, ax = plt.subplots(1, 1, dpi=ddpi)
    ax.plot(np.divide(frontierRanges, 1000), np.divide(frontierPayloads, 1000), color="blue")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("Range, km")
    ax.set_ylabel("Payload, tonnes")

    ax.grid()

    X = np.divide(contPayloadRangeFEPR[:,:,1], 1000)
    Y = np.divide(contPayloadRangeFEPR[:,:,0], 1000)
    Z = normalisedFEPR

    #ax.scatter(X, Y, marker="x", color="black")
    CS = ax.contour(X, 
                    Y,
                    Z,
                    levels=np.multiply(np.subtract((1.05, 1.10, 1.25, 1.50, 2.00, 3.00, 6.00), 1), 100),
                    colors="black")
    
    ax.scatter(frontierRanges[1]/1000, frontierPayloads[1]/1000, s=50, marker="o", color="black", edgecolors="white", linewidth=2, zorder=100)
    ax.annotate(f"Minimum FEPR = {minFEPR:.2f} J/kgm", (frontierRanges[1]/1000, frontierPayloads[1]/1000), zorder=200)

    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"+{s}\%" if plt.rcParams["text.usetex"] else f"+{s}%"

    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, manual=True)

    plt.show()