"""Assignment-style policy file.

Keep the entrypoint function signature and return style unchanged:
- def select_action(state)
- return HereAndNowActions
"""


def _policy_dummy(state):
    return 0.0, 0.0, 0


def _policy_threshold(state, t_low=20.0, t_high=24.0, p_max=3.0, h_high=70.0):
    t1 = float(state.get("T1", 21.0))
    t2 = float(state.get("T2", 21.0))
    h = float(state.get("H", 50.0))

    p1 = p_max if t1 < t_low else 0.0
    p2 = p_max if t2 < t_low else 0.0
    v = 1 if (h > h_high or t1 > t_high or t2 > t_high) else 0
    return p1, p2, v


def _policy_price_aware(state, cheap_price=3.5, p_max=3.0, h_high=70.0):
    t1 = float(state.get("T1", 21.0))
    t2 = float(state.get("T2", 21.0))
    h = float(state.get("H", 50.0))
    price = float(state.get("price_t", 4.0))

    if price <= cheap_price:
        p1 = p_max if t1 < 22.0 else 0.0
        p2 = p_max if t2 < 22.0 else 0.0
    else:
        p1 = p_max if t1 < 19.5 else 0.0
        p2 = p_max if t2 < 19.5 else 0.0

    v = 1 if h > h_high else 0
    return p1, p2, v


def _dispatch_policy(state,POLICY_TO_RUN):
    if POLICY_TO_RUN == "dummy":
        return _policy_dummy(state)
    if POLICY_TO_RUN == "threshold":
        return _policy_threshold(state)
    if POLICY_TO_RUN == "price_aware":
        return _policy_price_aware(state)
    raise ValueError("Unknown POLICY_TO_RUN. Use: dummy, threshold, price_aware")


def select_action(state):
    p1, p2, v = _dispatch_policy(state)

    HereAndNowActions = {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON": v,
    }

    return HereAndNowActions