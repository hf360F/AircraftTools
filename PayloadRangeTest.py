from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Performance  import payload_range
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Plots.Performance.Mission_Plots import *
import openvsp as vsp
import baseline as base

A320 = base.Baseline("Airbus A320-200ceo")

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # define the problem
    configs, analyses = full_setup()

    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()

    # mission analysis
    mission = analyses.missions
    results = mission.evaluate()

    # run payload diagram
    config = configs.base
    cruise_segment_tag = "cruise"
    reserves = 1750.
    payload_range_results = payload_range(config,mission,cruise_segment_tag,reserves)

    # plot the results
    plot_mission(results)

    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle_setup()
    vehicle = A320.suaveVehicle
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = mission

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.drag_coefficient_increment = 0.1000

    return analyses

def base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    #stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    #stability.geometry = vehicle
    #analyses.append(stability)

    # ------------------------------------------------------------------
    #  Propulsion Analysis
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = mission

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    """This is the full physical definition of the vehicle, and is designed to be independent of the
    analyses that are selected."""
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    #A320 = base.Baseline("Airbus A320-200ceo")
    vehicle = A320.suaveVehicle
    vehicle.tag = A320.dispName
    #A320.showVehicleGeom()
    #A320.report()

    #vehicle = SUAVE.Vehicle()
    #vehicle.tag = 'Boeing_737-800'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # The takeoff weight is used to determine the weight of the vehicle at the start of the mission
    vehicle.mass_properties.takeoff = vehicle.mass_properties.max_takeoff
    # Cargo weight typically feeds directly into weights output and does not affect the mission
    vehicle.mass_properties.cargo = 0   
    
    vehicle.mass_properties.max_fuel = vehicle.mass_properties.max_takeoff - vehicle.mass_properties.operating_empty
    vehicle.mass_properties.max_payload = vehicle.mass_properties.max_zero_fuel - vehicle.mass_properties.operating_empty

    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # Vehicle level parameters
    # Number of passengers, control settings, and accessories settings are used by the weights
    # methods
    vehicle.passengers = 180
    vehicle.systems.control = "fully powered" 
    vehicle.systems.accessories = "medium range"

    # ------------------------------------------------------------------
    #   Main Wing Control Surfaces
    # ------------------------------------------------------------------
    
    # Information in this section is used for high lift calculations and when conversion to AVL
    # is desired.
    
    # Deflections will typically be specified separately in individual vehicle configurations.
    
    flap = SUAVE.Components.Wings.Control_Surfaces.Flap() 
    flap.tag = "flap"
    flap.span_fraction_start = 0.20 
    flap.span_fraction_end = 0.70   
    flap.deflection = 0.0 * Units.degrees
    # Flap configuration types are used in computing maximum CL and noise
    flap.configuration_type = "single_slotted"
    flap.chord_fraction = 0.085   
    vehicle["wings"]["main_wing"].append_control_surface(flap)
        
    slat = SUAVE.Components.Wings.Control_Surfaces.Slat() 
    slat.tag = "slat" 
    slat.span_fraction_start = 0.04 
    slat.span_fraction_end = 0.92     
    slat.deflection = 0.0 * Units.degrees
    slat.chord_fractions = 0.12 	 
    vehicle["wings"]["main_wing"].append_control_surface(slat)
        
    aileron = SUAVE.Components.Wings.Control_Surfaces.Aileron() 
    aileron.tag = "aileron"
    aileron.span_fraction_start = 0.76 
    aileron.span_fraction_end = 0.97 
    aileron.deflection = 0.0 * Units.degrees
    aileron.chord_fraction = 0.35    
    vehicle["wings"]["main_wing"].append_control_surface(aileron) 

    # ------------------------------------------------------------------
    #   Nacelles
    # ------------------------------------------------------------------ 
    nacelle = SUAVE.Components.Nacelles.Nacelle()
    nacelle.tag = "nacelle_1"
    nacelle.length = 2.42
    nacelle.inlet_diameter = 1.73
    nacelle.diameter = 2.01
    nacelle.areas.wetted = 1.1*np.pi*nacelle.diameter*nacelle.length
    nacelle.origin = [[12.0, -11.51/2, -1.9]]
    nacelle.flow_through = True  
    nacelle_airfoil  = SUAVE.Components.Airfoils.Airfoil() 
    nacelle_airfoil.naca_4_series_airfoil = "2410"
    nacelle.append_airfoil(nacelle_airfoil)

    nacelle_2 = deepcopy(nacelle)
    nacelle_2.tag = "nacelle_2"
    nacelle_2.origin  = [[12.0, 11.51/2, -1.9]]
    
    vehicle.append_component(nacelle)  
    vehicle.append_component(nacelle_2)     
        
    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    # For some methods, the 'turbofan' tag is still necessary. This will be changed in the
    # future to allow arbitrary tags.
    turbofan.tag = "turbofan"
    
    # High-level setup
    turbofan.number_of_engines = 2
    turbofan.bypass_ratio = 5.4
    turbofan.origin = [nacelle.origin, nacelle_2.origin] * Units.meter

    # Establish the correct working fluid
    turbofan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    # Components use estimated efficiencies. Estimates by technology level can be
    # found in textbooks such as those by J.D. Mattingly
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # Converts freestream static to stagnation quantities
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = "ram"
    
    # add to the network
    turbofan.append(ram)

    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # Create component
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = "inlet_nozzle"
    
    # Specify performance
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio = 0.98
    
    # Add to network
    turbofan.append(inlet_nozzle)
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # Create component
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = "low_pressure_compressor"

    # Specify performance
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio = 1.14    
    
    # Add to network
    turbofan.append(compressor)
    
    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # Create component
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = "high_pressure_compressor"
    
    # Specify performance
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio  = 13.415    
    
    # Add to network
    turbofan.append(compressor)

    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine
    
    # Create component
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag = "low_pressure_turbine"
    
    # Specify performance
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # Add to network
    turbofan.append(turbine)
      
    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine
    
    # Create component
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag = "high_pressure_turbine"

    # Specify performance
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # Add to network
    turbofan.append(turbine)  
    
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # Create component    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = "combustor"
    
    # Specify performance
    combustor.efficiency  = 0.99 
    combustor.alphac = 1.0   
    combustor.turbine_inlet_temperature = 1450 # K
    combustor.pressure_ratio = 0.95
    combustor.fuel_data = SUAVE.Attributes.Propellants.Jet_A()    
    
    # Add to network
    turbofan.append(combustor)

    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # Create component
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = "core_nozzle"
    
    # Specify performance
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio = 0.99    
    
    # Add to network
    turbofan.append(nozzle)

    # ------------------------------------------------------------------
    #  Component 9 - Fan Nozzle
    
    # Create component
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = "fan_nozzle"

    # Specify performance
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio = 0.99    
    
    # Add to network
    turbofan.append(nozzle)
    
    # ------------------------------------------------------------------
    #  Component 10 - Fan
    
    # Create component
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag = "fan"

    # Specify performance
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio = 1.7    
    
    # Add to network
    turbofan.append(fan)
    
    # ------------------------------------------------------------------
    #  Component 11 - thrust (to compute the thrust)
    
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag = "compute_thrust"
 
    # Design thrust is used to determine mass flow at full throttle
    thrust.total_design = 2*24000 * Units.N #Newtons
    
    # Add to network
    turbofan.thrust = thrust
    
    # Design sizing conditions are also used to determine mass flow
    altitude = 39000.0*Units.ft
    mach_number = 0.78     

    # Determine turbofan behavior at the design condition
    turbofan_sizing(turbofan,mach_number,altitude)   
    
    # Add turbofan network to the vehicle 
    vehicle.append_component(turbofan)      
    
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    SUAVE.Plots.Geometry.plot_vehicle(vehicle, plot_control_points=False)

    A320.suaveVehicle = vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    configs.append(config)


    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'takeoff'
    config.wings['main_wing'].control_surfaces.flap.deflection  = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.V2_VS_ratio = 1.21
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    config.wings['main_wing'].control_surfaces.flap.deflection  = 30. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.Vref_VS_ratio = 1.23
    configs.append(config)   

    # done!
    return configs

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'Airbus A320-200 test mission'

    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()

    # ------------------------------------------------------------------
    #   First Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.takeoff )

    # define segment attributes
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 190.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Cruise Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 447. * Units.knots 
    segment.distance   = 2100. * Units.nmi

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end = 8.0   * Units.km
    segment.air_speed    = 220.0 * Units['m/s']
    segment.descent_rate = 4.5   * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.landing )

    # segment attributes
    segment.altitude_end = 6.0   * Units.km
    segment.air_speed    = 195.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.landing )

    # segment attributes
    segment.altitude_end = 4.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Fourth Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_4"

    segment.analyses.extend( analyses.landing )

    segment.altitude_end = 2.0   * Units.km
    segment.air_speed    = 150.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']


    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Fifth Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_5"

    segment.analyses.extend( analyses.landing )

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 3.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete
    # ------------------------------------------------------------------

    return mission

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='bo-'):

    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style)
    
    # Plot Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results, line_style)    
    
    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results, line_style)    


    return

if __name__ == '__main__':
    main()
    plt.show()