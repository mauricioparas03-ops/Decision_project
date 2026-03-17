from pyomo.environ import *

def select_action(state):
    
    HereAndNowActions = {
    "HeatPowerRoom1" : 0, 
    "HeatPowerRoom2" : 0, 
    "VentilationON" : 0 
    }

    return HereAndNowActions