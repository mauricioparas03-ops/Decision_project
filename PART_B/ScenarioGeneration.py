
price_dict   = {}
occ_dict     = {}

for s in range(n_scenarios):
    p_cur,  p_prev  = price_now,  price_prev
    o1_cur, o2_cur  = occ_r1_now, occ_r2_now

    for t in range(horizon):
        p_next           = price_model(p_cur, p_prev)
        o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

        price_dict[t, s]   = p_next
        occ_dict[1, t, s]  = o1_next
        occ_dict[2, t, s]  = o2_next

        p_prev, p_cur   = p_cur,  p_next
        o1_cur, o2_cur  = o1_next, o2_next

