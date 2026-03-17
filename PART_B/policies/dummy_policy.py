from pyomo.environ import *

def select_action(self, state):
    """
    Dummy policy: never turns on the ventilation nor any heater actively.
    Everything is left up to the overrule controllers.
    """
    HereAndNowActions = {
        "HeatPowerRoom1": 0.0,  # no heating room 1
        "HeatPowerRoom2": 0.0,  # no heating room 2
        "Ventilation": 0      # no ventilation
    }
    
    return HereAndNowActions