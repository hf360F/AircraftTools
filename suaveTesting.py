import openvsp as vsp
import baseline as base
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

## Derived from 737 example
def vehicleSetup():
    """Define baseline vehicle.
    """

    vehicle = SUAVE.Vehicle()
    vehicle.tag = "Boeing_737-800"

    # Masses
    vehicle.mass_properties.max_takeoff = 79015.8 * Units.kilogram
    vehicle.mass_properties.takeoff = 79015.8 * Units.kilogram
    vehicle.mass_properties.operating_empty = 62746.4 * Units.kilogram 
    vehicle.mass_properties.max_zero_fuel = 62732.0 * Units.kilogram
    vehicle.mass_properties.cargo = 10000.0 * Units.kilogram   

    # Flight envelope
    vehicle.envelope.ultimate_load = 3.75
    vehicle.envelope.limit_load = 2.5
    
    # Reference parameters
    vehicle.reference_area = 124.862 * Units["meters**2"] # Main wing area
    vehicle.passengers = 170
    vehicle.systems.control = "fully powered"
    vehicle.systems.accessories = "medium range"

    # Gear configuration
    landingGear = SUAVE.Components.Landing_Gear.Landing_Gear()
    landingGear.tag = "main_gear"
    landingGear.main_tire_diameter = 1.12000 * Units.m
    landingGear.nose_tire_diameter = 0.6858 * Units.m
    landingGear.main_strut_length  = 1.8 * Units.m
    landingGear.nose_strut_length  = 1.3 * Units.m
    landingGear.main_units  = 2    # Number of main landing gear
    landingGear.nose_units  = 1    # Number of nose landing gear
    landingGear.main_wheels = 2    # Number of wheels on the main landing gear
    landingGear.nose_wheels = 2    # Number of wheels on the nose landing gear      
    vehicle.landing_gear = landingGear

    # Wing definition
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = "main_wing"
    wing.aspect_ratio = 10.18
    wing.sweeps.quarter_chord = 25 * Units.deg
    wing.thickness_to_chord = 0.1
    wing.taper = 0.1
    wing.spans.projected = 34.32 * Units.meter
    wing.chords.root = 7.760 * Units.meter
    wing.chords.tip = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    wing.areas.reference = 124.862 * Units["meters**2"]  
    wing.twists.root = 4.0 * Units.degrees
    wing.twists.tip = 0.0 * Units.degrees
    wing.origin = [[13.61, 0, -1.27]] * Units.meter
    wing.vertical = False
    wing.symmetric = True
    # The high lift flag controls aspects of maximum lift coefficient calculations
    wing.high_lift = True
    # The dynamic pressure ratio is used in stability calculations
    wing.dynamic_pressure_ratio  = 1.0

    # Surfaces
    flap = SUAVE.Components.Wings.Control_Surfaces.Flap() 
    flap.tag = "flap"
    flap.span_fraction_start = 0.20 
    flap.span_fraction_end = 0.70   
    flap.deflection = 0.0 * Units.degrees
    flap.configuration_type = "double_slotted"
    flap.chord_fraction = 0.30   
    wing.append_control_surface(flap)   
        
    slat = SUAVE.Components.Wings.Control_Surfaces.Slat() 
    slat.tag = "slat"
    slat.span_fraction_start = 0.324 
    slat.span_fraction_end = 0.963     
    slat.deflection = 0.0 * Units.degrees
    slat.chord_fraction = 0.1  	 
    wing.append_control_surface(slat)  
        
    aileron = SUAVE.Components.Wings.Control_Surfaces.Aileron() 
    aileron.tag = "aileron" 
    aileron.span_fraction_start = 0.7 
    aileron.span_fraction_end = 0.963 
    aileron.deflection = 0.0 * Units.degrees
    aileron.chord_fraction = 0.16    
    wing.append_control_surface(aileron)    
    
    vehicle.append_component(wing)

    # Tail
    wing = SUAVE.Components.Wings.Horizontal_Tail()
    wing.tag = "horizontal_stabilizer"
    wing.aspect_ratio = 6.16     
    wing.sweeps.quarter_chord = 40.0 * Units.deg
    wing.thickness_to_chord = 0.08
    wing.taper = 0.2
    wing.spans.projected = 14.2 * Units.meter
    wing.chords.root = 4.7  * Units.meter
    wing.chords.tip = 0.955 * Units.meter
    wing.chords.mean_aerodynamic = 3.0  * Units.meter
    wing.areas.reference = 32.488   * Units["meters**2"]  
    wing.twists.root = 3.0 * Units.degrees
    wing.twists.tip = 3.0 * Units.degrees  
    wing.origin = [[32.83 * Units.meter, 0 , 1.14 * Units.meter]]
    wing.vertical = False 
    wing.symmetric = True
    wing.dynamic_pressure_ratio = 0.9
    vehicle.append_component(wing)

    wing = SUAVE.Components.Wings.Vertical_Tail()
    wing.tag = 'vertical_stabilizer'
    wing.aspect_ratio = 1.91
    wing.sweeps.quarter_chord = 25. * Units.deg
    wing.thickness_to_chord = 0.08
    wing.taper = 0.25
    wing.spans.projected = 7.777 * Units.meter
    wing.chords.root = 8.19  * Units.meter
    wing.chords.tip = 0.95  * Units.meter
    wing.chords.mean_aerodynamic = 4.0   * Units.meter
    wing.areas.reference = 27.316 * Units['meters**2']  
    wing.twists.root = 0.0 * Units.degrees
    wing.twists.tip = 0.0 * Units.degrees  
    wing.origin = [[28.79 * Units.meter, 0, 1.54 * Units.meter]] # meters
    wing.vertical = True 
    wing.symmetric = False
    wing.t_tail = False
    wing.dynamic_pressure_ratio = 1.0
    vehicle.append_component(wing)

    # Fuselage
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = "Fuselage"
    
    fuselage.number_coach_seats = vehicle.passengers
    fuselage.seats_abreast = 6
    fuselage.seat_pitch = 1 * Units.meter
    fuselage.fineness.nose = 1.6
    fuselage.fineness.tail = 2.0
    fuselage.lengths.nose = 6.4   * Units.meter
    fuselage.lengths.tail = 8.0   * Units.meter
    fuselage.lengths.total = 38.02 * Units.meter
    fuselage.lengths.fore_space = 6. * Units.meter
    fuselage.lengths.aft_space = 5. * Units.meter
    fuselage.width = 3.74 * Units.meter
    fuselage.heights.maximum = 3.74 * Units.meter
    fuselage.effective_diameter = 3.74 * Units.meter
    fuselage.areas.side_projected = 142.1948 * Units["meters**2"] 
    fuselage.areas.wetted = 446.718 * Units["meters**2"] 
    fuselage.areas.front_projected = 12.57 * Units["meters**2"] 
    fuselage.differential_pressure = 5.0e4 * Units.pascal
    
    fuselage.heights.at_quarter_length          = 3.74 * Units.meter
    fuselage.heights.at_three_quarters_length   = 3.65 * Units.meter
    fuselage.heights.at_wing_root_quarter_chord = 3.74 * Units.meter
    
    vehicle.append_component(fuselage)

    # Engines
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = "turbofan"
    
    turbofan.number_of_engines = 2
    turbofan.bypass_ratio = 5.4
    turbofan.engine_length = 2.71 * Units.meter
    turbofan.nacelle_diameter = 2.05 * Units.meter
    turbofan.origin = [[13.72, 4.86,-1.9],[13.72, -4.86,-1.9]] * Units.meter
    
    turbofan.areas.wetted  = 1.1*np.pi*turbofan.nacelle_diameter*turbofan.engine_length
    
    turbofan.working_fluid = SUAVE.Attributes.Gases.Air()

    # Thrust component
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'

    # Design thrust is used to determine mass flow at full throttle
    thrust.total_design = 2*24000. * Units.N

    # Design sizing conditions are also used to determine mass flow
    altitude      = 35000.0*Units.ft
    mach_number   = 0.78 

    # Add to network
    turbofan.thrust = thrust

    return vehicle

def configsSetup(vehicle):
    """Set up vehicle configurations for use in different mission phases.

    Args:
        vehicle (_type_): _description_
    """

    configs = SUAVE.Components.Configs.Config.Container()

    baseConfig = SUAVE.Components.Configs.Config(vehicle)
    baseConfig.tag = "Base"
    configs.append(baseConfig)

    config = SUAVE.Components.Configs.Config(baseConfig)
    config.tag = "Cruise"
    configs.append(config)

    config = SUAVE.Components.Configs.Config(baseConfig)
    config.tag = "Takeoff"
    config.wings["main_wing"].control_surfaces.flap.deflection = 20. * Units.deg
    config.wings["main_wing"].control_surfaces.slat.deflection = 25. * Units.deg
    config.max_lift_coefficient_factor = 1
    configs.append(config)

    config = SUAVE.Components.Configs.Config(baseConfig)
    config.tag = "Landing"
    config.wings["main_wing"].control_surfaces.flap.deflection = 30. * Units.deg
    config.wings["main_wing"].control_surfaces.slat.deflection = 25. * Units.deg  
    config.max_lift_coefficient_factor = 1
    configs.append(config)

    return configs

def simpleSizing(configs):
    """Apply basic sizing relations and create landing configuration.

    Args:
        configs (_type_): _description_
    """

    base = configs.base
    base.pull_base()

    # Adjust zero fuel weight for base configuration for landing
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff

    # Estimate wing areas
    for wing in base.wings:
        wing.areas.wetted = 2.0 * wing.areas.reference
        wing.areas.exposed = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted

    # Store changes
    base.store_diff()

    # Landing configuration
    landing = configs.landing
    landing.pull_base
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff
    landing.store_diff()

    return

def analysesSetup(configs):
    """Create analyses for each aircraft configuration.

    Args:
        configs (_type_): _description_
    """

    analyses = SUAVE.Analyses.Analysis.Container()

    for tag, config in configs.items():
        analysis = baseAnalysis(config)
        analyses[tag] = analysis

    return analyses

def baseAnalysis(vehicle):
    """Baseline set of analyses to be conducted for vehicle.

    Args:
        vehicle (_type_): _description_
    """

    analyses = SUAVE.Analyses.Vehicle()

    # Mass
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # Aerodynamics
    aero = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aero.geometry = vehicle
    analyses.append(aero)

    # Stability
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    # Energy network
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # Planet
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # Atmosphere
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo.features.planet = planet.features
    analyses.append(atmo)

    return analyses

def missionSetup(analyses): 
    """Defines baseline mission flown to examine performance.

    Args:
        analyses (_type_): _description_
    """

    # Initialise
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = "Flight mission"

    # Airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude = 0.0 * Units.ft
    airport.delta_isa = 0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.airport = airport
    
    # Segments module unpack
    Segments = SUAVE.Analyses.Mission.Segments

    # Base segment
    baseSegment = Segments.Segment()

    # Climbout at constant speed and constant climb rate
    segment = Segments.Climb.Constant_Speed_Constant_Rate(baseSegment)
    segment.tag = "Climbout"
    segment.analyses.extend(analyses.takeoff)
    segment.altitude_start = 0.0 * Units.km
    segment.altitude_end = 3.0 * Units.km
    segment.air_speed = 125.0 * Units["m/s"]
    segment.climb_rate = 6.0 * Units["m/s"]
    mission.append_segment(segment)

    return mission

def missionsSetup(baseMission):
    """Allows multiple missions to be flown.

    Args:
        baseMission (_type_): _description_
    """

    missions = SUAVE.Analyses.Mission.Mission.Container()

    missions.base = baseMission

    return missions

def plotMission(results, linestyle="bo-"):
    """Plots mission analysis results and saves to .png files.

    Args:
        results (_type_): _description_
        linestyle (str, optional): _description_. Defaults to "bo-".
    """

    plot_flight_conditions(results, linestyle)    

def fullSetup():
    """Get baseline vehicle, create modifications for different configurations,
    and create mission and analyses for these configurations.

    Returns:
        _type_: _description_
        _type_: _description_
    """

    # Collect baseline vehicle data and changes when using different configuration settings
    vehicle = vehicleSetup()
    configs = configsSetup(vehicle)

    # Get the analyses to be used when different configurations are evaluated
    configs_analyses = analysesSetup(configs)

    # Create the mission that will be flown
    mission = missionSetup(configs_analyses)
    missions_analyses = missionsSetup(mission)

    # Add the analyses to the proper containers
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

def main():
    # Extract vehicle configurations and the analysis settings that go with them
    configs, analyses = fullSetup()

    # Size each of the configurations according to a given set of geometry relations
    simpleSizing(configs)

    # Perform operations needed to make the configurations and analyses usable in the mission
    configs.finalize()
    analyses.finalize()

    # Determine the vehicle weight breakdown (independent of mission fuel usage)
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()      

    # Performance a mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # Plot all mission results, including items such as altitude profile and L/D
    plotMission(results)

    return

main()