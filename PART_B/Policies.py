class Policies:
    @staticmethod
    def select_action(state):
        
        
        
        ### Here goes your code
        
        HereAndNowActions = {
        "HeatPowerRoom1" : 0, #replace 0 with your choice
        "HeatPowerRoom2" : 0, #replace 0 with your choice
        "VentilationON" : 0 #binary. replace 0 with 1 if your choice is ON
        }
        
        return HereAndNowActions

    #print(select_action(1))