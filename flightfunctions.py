import numpy as np
import matplotlib.pyplot as plt

import SUAVE
from SUAVE.Core import Data, Units
import baseline as base

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

def flightMission(Aircraft, range, fuel, payload, climbType, cruiseAlt):

    ## Vehicle configuration
    vehicle = Aircraft.suaveVehicle
    if fuel > vehicle.mass_properties.max_fuel:
        print(f"Fuel guess {fuel:.0f} kg exceeds maximum {vehicle.mass_properties.max_fuel} kg for {vehicle.tag}. Clamping to max.")
        fuel = vehicle.mass_properties.max_fuel
    if payload > vehicle.mass_properties.max_payload:
        print(f"Payload {payload:.0f} kg exceeds maximum {vehicle.mass_properties.max_payload} kg for {vehicle.tag}. Clamping to max.")
        payload = vehicle.mass_properties.max_payload
    if (payload+fuel+vehicle.mass_properties.operating_empty) > vehicle.mass_properties.max_takeoff:
        print(f"Take-off mass {(payload+fuel+vehicle.mass_properties.operating_empty):.0f} kg exceeds "\
              f"maximum {vehicle.mass_properties.max_takeoff:.0f} kg for {vehicle.tag}. Dumping fuel to clamp to max.")
        fuel -= (payload+fuel+vehicle.mass_properties.operating_empty) - vehicle.mass_properties.max_takeoff 

    # Guess fuel of 0.5 of max allowable?

    vehicle.mass_properties.payload = payload

    fuelRelTol = 1E-5
    exponent = 1.05
    fuelRatio = 10
    maxIters = 10

    def base_analysis(vehicle):
        analyses = SUAVE.Analyses.Vehicle()

        weights = SUAVE.Analyses.Weights.Weights_Transport()
        weights.vehicle = vehicle
        analyses.append(weights)

        aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
        aerodynamics.geometry = vehicle
        analyses.append(aerodynamics)

        stability = SUAVE.Analyses.Stability.Fidelity_Zero()
        stability.geometry = vehicle
        analyses.append(stability)

        energy = SUAVE.Analyses.Energy.Energy()
        energy.network = vehicle.networks
        analyses.append(energy)

        planet = SUAVE.Analyses.Planets.Planet()
        analyses.append(planet)

        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features
        analyses.append(atmosphere) 

        return analyses

    def analyses_setup(configs):
        analyses = SUAVE.Analyses.Analysis.Container()

        for tag, config in configs.items():
            analysis = base_analysis(config)
            analyses[tag] = analysis

        return analyses

    def fly():
        mission = SUAVE.Analyses.Mission.Sequential_Segments()
        mission.tag = "Flight Mission"

        airport = SUAVE.Attributes.Airports.Airport()
        airport.altitude = 0.0 * Units.m
        airport.delta_isa = 0.0
        airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()  
        mission.airport = airport  

        Segments = SUAVE.Analyses.Mission.Segments
        baseSegment = Segments.Segment()

        configs = SUAVE.Components.Configs.Config.Container()

        climbTypes = ("CruiseOnly")
        if climbType not in climbTypes:
            raise ValueError(f"Cruise type '{climbType}' unsupported. Must be one of {climbTypes}")
        elif climbType is "CruiseOnly":
            baseConfig = SUAVE.Components.Configs.Config(vehicle)
            baseConfig.tag = "base"
            configs.append(baseConfig)

            config = SUAVE.Components.Configs.Config(baseConfig)
            config.tag = "cruise"
            configs.append(config)

            configs_analyses = analyses_setup(configs)

            # Define mission
            segment = Segments.Cruise.Constant_Speed_Constant_Altitude(baseSegment)
            segment.tag = "cruise"
            segment.analyses.extend(configs_analyses.cruise)
            cruiseConds = configs_analyses.base.atmosphere.compute_values(cruiseAlt)
            segment.air_speed = Aircraft.cruise_mach * cruiseConds.speed_of_sound
            segment.altitude = cruiseAlt
            segment.distance = range
            mission.append_segment(segment)

        # Finalise
        missions_analyses = SUAVE.Analyses.Mission.Mission.Container()
        missions_analyses.base = mission

        analyses = SUAVE.Analyses.Analysis.Container()
        analyses.configs = configs_analyses
        analyses.missions = missions_analyses

        base = configs.base
        base.pull_base()
        base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 
        
        for wing in base.wings:
            wing.areas.wetted   = 2.0 * wing.areas.reference
            wing.areas.exposed  = 0.8 * wing.areas.wetted
            wing.areas.affected = 0.6 * wing.areas.wetted
        base.store_diff()

        configs.finalize()
        analyses.finalize()

        weights = analyses.configs.base.weights
        breakdown = weights.evaluate()

        # Fly
        mission = analyses.missions.base
        results = mission.evaluate()

        return results


    results = fly()

    return results

def fixedRangeMission(Aircraft, range, payload, climbType, cruiseAlt):
    vehicle = Aircraft.suaveVehicle
    # Crude initial guess
    fuel = 0.5*np.min((vehicle.mass_properties.max_fuel,
                      vehicle.mass_properties.max_takeoff - (vehicle.mass_properties.operating_empty + payload)))
    
    i = 0
    fuelRelTol = 1E-5
    fuelRatio = 0
    exponent = 1.05
    maxIters = 10

    while fuelRatio > 1+fuelRelTol or fuelRatio < 1-fuelRelTol:
        if i == 0:
            fuelLoad = fuel
        elif i == maxIters:
            raise ValueError(f"ITERATION COUNT EXCEEDS MAX ({maxIters})")

        vehicle.mass_properties.fuel = fuelLoad
        vehicle.mass_properties.takeoff = vehicle.mass_properties.operating_empty + fuelLoad + payload

        if vehicle.mass_properties.fuel > vehicle.mass_properties.max_fuel:
            raise ValueError(f"Next fuel load {vehicle.mass_properties.fuel:.1f} kg exceeds max {vehicle.mass_properties.max_fuel:.1f} kg. Range may be unreachable.")
        if vehicle.mass_properties.takeoff > vehicle.mass_properties.max_takeoff:
            raise ValueError(f"Next takeoff weight {vehicle.mass_properties.takeoff:.1f} kg exceeds max {vehicle.mass_properties.max_takeoff:.1f} kg. Range may be unreachable.")

        results = flightMission(Aircraft, range, fuel, payload, climbType, cruiseAlt)

        initialMass = results.segments[0].conditions.weights.total_mass[0,0]
        finalMass = results.segments[-1].conditions.weights.total_mass[-1,0]

        fuelBurn = initialMass - finalMass
        fuelRatio = fuelBurn/fuelLoad

        fuelLoadOld = fuelLoad
        fuelLoad = fuelLoad * (fuelRatio**exponent)

        print(f"FLIGHT ITERATION: {i}")
        print(f"LOADED {fuelLoadOld:.1f} kg, BURNED {fuelBurn:.1f} kg, RATIO = {fuelRatio:.4f}, NEXT LOAD {fuelLoad:.1f} kg")
        i += 1

    print(f"CONVERGED WITH FUEL LOAD {fuelLoad:.1f} kg")

    """
    resultsSummary = dict.fromkeys(["times",
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
    """

    return results

def fixedFuelMission(Aircraft, fuel, payload, climbType, cruiseAlt):
    vehicle = Aircraft.suaveVehicle
    vehicle.mass_properties.fuel = fuel
    vehicle.mass_properties.takeoff = vehicle.mass_properties.operating_empty + fuel + payload

   # Crude initial guess - likely to be +- 1 OOM
    range = 2000 * Units.km
    
    i = 0
    fuelUseRelTol = 1E-5
    fuelUseFraction = 0
    exponent = 1.05
    maxIters = 10

    while fuelUseFraction > 1+fuelUseRelTol or fuelUseFraction < 1-fuelUseRelTol:
        if i == 0:
            newRange = range
        elif i == maxIters:
            raise ValueError(f"ITERATION COUNT EXCEEDS MAX ({maxIters})")

        results = flightMission(Aircraft, newRange, fuel, payload, climbType, cruiseAlt)

        initialMass = results.segments[0].conditions.weights.total_mass[0,0]
        finalMass = results.segments[-1].conditions.weights.total_mass[-1,0]

        fuelBurn = initialMass - finalMass
        fuelUseFraction = fuelBurn/fuel

        oldRange = results.segments[-1].distance
        newRange = oldRange * ((1/fuelUseFraction) ** exponent)

        print(f"FLIGHT ITERATION: {i}")
        print(f"FLEW {oldRange/1000:.1f} km, BURNED {fuelBurn:.1f} kg, FUEL BURN FRACTION = {fuelUseFraction:.4f}, NEXT RANGE {newRange/1000:.1f} km")
        i += 1

    print(f"CONVERGED WITH RANGE {newRange/1000:.1f} km")

    return results