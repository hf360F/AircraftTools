import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import SUAVE
from SUAVE.Core import Data, Units
import baseline as base

def aeroSweep(vehicle, alphas, machs, altitude, deltaT=0):
    """Perform a Mach/alpha sweep on a vehicle.

    Args:
        vehicle (SUAVE.Vehicle.Vehicle): SUAVE vehicle to perform analysis on.
        alphas (numpy.ndarray): List of angles of attack, used to vary lift coefficient.
        machs (numpy.ndarray): List of Mach numbers at which to run alpha sweep.
        altitude (float): Freestream altitude, m.
        deltaT (float, optional): ISA temperature offset. Defaults to 0.

    Returns:
        dict: Dictionary of results of interest. Each entry's value is a 2D array, len(machs) x len(alphas).
    """

    # Results list
    grid = np.zeros((len(machs), len(alphas)))
    results = {"totalLift": np.zeros_like(grid),
               "totalDrag": np.zeros_like(grid),
               "liftToDrag": np.zeros_like(grid),
               "parasiticDragTotal": np.zeros_like(grid),
               "parasiticDragFull": np.ndarray((len(machs), len(alphas)), dtype=Data),
               "inducedDrag": np.zeros_like(grid),
               "compDrag": np.zeros_like(grid),
               "miscDrag": np.zeros_like(grid)}
    results2 = [] 

    # Approximate wing areas
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted 

    aero = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aero.geometry = vehicle
    aero.initialize()

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
    
    i = 0
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
        results2.append(result)

        results["totalLift"][i] = [value[0] for value in result.lift.total]
        results["totalDrag"][i] = [value[0] for value in result.drag.total[0]]
        results["liftToDrag"][i] = np.divide(results["totalLift"][i], results["totalDrag"][i])
        results["parasiticDragTotal"][i] = [value[0] for value in result.drag.parasite.total[0]]
        results["parasiticDragFull"][i] = result.drag.parasite
        results["inducedDrag"][i] = [value[0] for value in result.drag.induced[0]]
        results["compDrag"][i] = [value[0] for value in result.drag.compressibility.total]
        results["miscDrag"][i] = [value[0] for value in result.drag.miscellaneous]
        
        i += 1

    results["machs"] = machs
    results["alphas"] = alphas

    return results

def flightEndurance(Aircraft, payload, fuel, holdingAltitude=1500/3.281, holdingIAS=230/1.944):
    """Estimate the endurance of the vehicle with given payload and initial fuel at specified conditions.

    Args:
        Aircraft (Container): Baseline or Derivative containing SUAVE.Vehicle.Vehicle object.
        payload (float): Payload, kg.
        fuel (float): Initial fuel, kg.
        holdingAltitude (float): Altitude, m. Defaults to 1500 ft (see ICAO Annex 6).
        holdingIAS (float): Indicated airspeed, m/s. Defaults to 230 kts, standard ICAO limit at default altitude.

    Returns:
        float: Estimated endurance, s.
    """

    vehicle = Aircraft.suaveVehicle
    averageMass = vehicle.mass_properties.operating_empty + payload + fuel/2

    # Atmospheric conditions at holding altitude
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo.features.planet = SUAVE.Analyses.Planets.Planet().features

    g = atmo.features.planet.sea_level_gravity
    cp = atmo.fluid_properties.specific_heat_capacity
    gamma = cp / (cp - atmo.fluid_properties.gas_specific_constant)

    holdingConds = atmo.compute_values(altitude=holdingAltitude, temperature_deviation=0)

    rho = holdingConds.density[:,None][0]
    a = holdingConds.speed_of_sound[:,None][0]
    mu = holdingConds.dynamic_viscosity[:,None][0]
    p = holdingConds.pressure[:,None][0]
    T = holdingConds.temperature[:,None][0]

    holdingTAS = holdingIAS * np.sqrt(1.225/rho)
    holdingMach = holdingTAS/a

    # Average lift coefficient
    holdingCl = averageMass*g/(vehicle.reference_area*0.5*rho*(holdingTAS**2))

    # Aero analysis - need to sweep alpha to match Cl, then lookup L/D at this Cl
    results = aeroSweep(vehicle = Aircraft.suaveVehicle,
                        alphas = np.linspace(-1*np.pi/180, 6*np.pi/180, 1000),
                        machs = np.array([holdingMach[0,0]]),
                        altitude = holdingAltitude)
    closestClIndex = np.argmin(np.abs(np.subtract(results["totalLift"][0], holdingCl)))
    holdingLD = results["liftToDrag"][0,closestClIndex]

    holdingThrust = averageMass*g/holdingLD # Level flight, no acceleration

    # Initialise state object needed for obtaining turbofan SFC
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    state.conditions.freestream.isentropic_expansion_factor = gamma
    state.conditions.freestream.specific_heat_at_constant_pressure = cp

    state.conditions.freestream.pressure = p
    state.conditions.freestream.temperature = T
    state.conditions.freestream.speed_of_sound = a

    state.conditions.freestream.velocity = holdingMach[0,0]*a
    state.conditions.freestream.mach_number = holdingMach[0,0]

    state.conditions.freestream.gravity = g

    state.conditions.aerodynamics.angle_of_attack = results["alphas"][closestClIndex]
    state.conditions.propulsion.throttle = 0.0

    def thrustError(throttle):
        state.conditions.propulsion.throttle = throttle
        thrustResult = vehicle.networks.turbofan.evaluate_thrust(state)
        return np.abs(holdingThrust - thrustResult.thrust_force_vector[0,0])
    
    holdingThrottle = optimize.minimize_scalar(thrustError, bounds=(0, 1)).x
    state.conditions.propulsion.throttle = holdingThrottle
    holdingThrustResult = vehicle.networks.turbofan.evaluate_thrust(state)

    holdingFuelRate = holdingThrustResult.vehicle_mass_rate[0,0]

    t = fuel/holdingFuelRate

    return t

def ICAOreserve(Aircraft, payload, fuel):
    altitude = 1500/3.281 # ft
    IAS = 230/1.944 # kts

    endurance = flightEndurance(Aircraft, payload, fuel, holdingAltitude=altitude, holdingIAS=IAS)

    contingency_reserve = np.max((fuel*5*60/endurance, 0.05*fuel)) # Max of 5 minutes holding endurance or 5% of trip fuel
    alternate_reserve = fuel*15*60/endurance # 15 minutes holding endurance when no destination alternate aerodrome is specified
    final_reserve = fuel*30*60/endurance

    return contingency_reserve + alternate_reserve + final_reserve

def climbDescentStages(Aircraft, endAltitude):
    """Identify the climb and descent stages for a flight with a maximum given altitude.

    Args:
        Aircraft (Container): Baseline or Derivative containing climb and descent stage information.
        endAltitude (float): Altitude at end of climb and start of descent, m.

    Raises:
        ValueError: Climb altitude exceeds end altitude of final climb segment.

    Returns:
        (list): List of list of climb segment EASs, climb segment Machs, final climb segment index, climb segment climb rates, climb segment end altitudes
    """
    endAltitude *= 3.281 # Convert to ft

    if endAltitude > Aircraft.climbEndAlts[-1]:
        raise ValueError(f"Specified climb altitude {endAltitude:.1f} ft exceeds altitude at end of last climb segment ({Aircraft.climbEndAlts[-1]:.1f} ft)")

    # For a given max flight level guess -
    # identify the climb and descent stages
    # Fly these stages with cruise segment = 0 km 

    # Find climb and descent stages for this guess
    i = 0
    for EndAlt in Aircraft.climbEndAlts:
        if EndAlt > endAltitude:
            climbIndex = i + 1
            break
        else:
            i += 1
    if climbIndex > len(Aircraft.climbEASs):
        climbEASs = Aircraft.climbEASs
        climbMachs = Aircraft.climbMachs
    else:
        climbEASs = Aircraft.climbEASs[:climbIndex]
        climbMachs = []
    climbRates = Aircraft.climbRates[:climbIndex]
    climbEndAltitudes = Aircraft.climbEndAlts[:climbIndex]

    i = 0
    for EndAlt in Aircraft.descentEndAlts:
        if EndAlt < endAltitude:
            descentIndex = i
            break
        else:
            i += 1
    # Flight descent stages = Aircraft descent stages - descentIndex
    if Aircraft.descentStages - descentIndex > len(Aircraft.descentEASs):
        descentEASs = Aircraft.descentEASs
        descentMachs = Aircraft.descentMachs[-(Aircraft.descentStages - descentIndex - len(Aircraft.descentEASs)):]
    else:
        descentEASs = Aircraft.descentEASs[descentIndex-1:]
        descentMachs = []
    descentRates = Aircraft.descentRates[descentIndex:]
    descentEndAltitudes = Aircraft.descentEndAlts[descentIndex:]

    return climbEASs, climbMachs, climbIndex, climbRates, climbEndAltitudes, descentMachs, descentEASs, descentRates, descentEndAltitudes


def flightMission(Aircraft, cruiseRange, fuel, payload, climbType, cruiseAlt):
    """Fly an aircraft for one mission. Fuel burn can exceed allocated fuel load.

    Args:
        Aircraft (Container): Baseline or Derivative containing SUAVE.Vehicle.Vehicle object.
        range (float): Total flight range, m.
        fuel (float): Aircraft fuel load, kg.
        payload (float): Aircraft payload, kg.
        climbType (string): Type of aircraft climb. Must be one of 'cruiseOnly'.
        cruiseAlt (float): Cruise altitude, m.

    Raises:
        ValueError: Climb type not supported.

    Returns:
        SUAVE.Core.Data.Data: SUAVE flight result.
    """

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

        climbTypes = ("CruiseOnly", "AircraftDefined", "ClimbDescentOnly")
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
    """Fly an aircraft a fixed range with a given payload, iterate to find required fuel load.

    Args:
        Aircraft (Container): Baseline or Derivative containing SUAVE.Vehicle.Vehicle object.
        range (float): Target toal range, m.
        payload (float): Aircraft payload, kg.
        climbType (string): Passed to flightFunctions.flightMission().
        cruiseAlt (float): Passed to flightFunctions.flightMission().

    Raises:
        ValueError: Unable to converge on target range due to aircraft fuel capacity limit.
        ValueError: Unable to converge on target range due to aircraft MTOW limit.
        ValueError: Iteration count exceeded.

    Returns:
        _type_: _description_
    """
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

    return results

def fixedFuelMission(Aircraft, fuel, payload, climbType, cruiseAlt):
    """Fly an aircraft to range exhausting all given fuel with constant payload by iterating cruise range.

    Args:
        Aircraft (Container): Baseline or Derivative containing SUAVE.Vehicle.Vehicle object.
        fuel (float): Aircraft fuel load, kg.
        payload (float): Aircraft payload, kg.
        climbType (string): Passed to flightFunctions.flightMission().
        cruiseAlt (float): Passed to flightFunctions.flightMission().

    Raises:
        ValueError: Iteration count exceeded.
    """

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