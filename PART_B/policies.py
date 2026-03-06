class DummyPolicy:

    def select_action(self, state):
        """
        Dummy policy: never turns on the ventilation nor any heater actively.
        Everything is left up to the overrule controllers.
        """
        HereAndNowActions = {
            "p1": 0.0,  # no heating room 1
            "p2": 0.0,  # no heating room 2
            "v": 0      # no ventilation
        }
        
        return HereAndNowActions