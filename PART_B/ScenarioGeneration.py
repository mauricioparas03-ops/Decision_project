
"""
Scenario generation and clustering for PART_B.

This script generates Monte Carlo scenarios for price and room occupancy,
clusters them with K-Means into 10 representative scenarios, and saves the
clustered trajectories to a CSV file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data


RAW_SCENARIOS = 50
N_CENTROIDS = 10
RANDOM_SEED = 42


def generate_scenarios(price_now, price_prev, occ_r1_now, occ_r2_now, horizon, n_scenarios):
    price_dict = {}
    occ_dict = {}

    for s in range(n_scenarios):
        p_cur, p_prev = price_now, price_prev
        o1_cur, o2_cur = occ_r1_now, occ_r2_now

        for t in range(horizon):
            p_next = price_model(p_cur, p_prev)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

            price_dict[t, s] = float(p_next)
            occ_dict[1, t, s] = float(o1_next)
            occ_dict[2, t, s] = float(o2_next)

            p_prev, p_cur = p_cur, p_next
            o1_cur, o2_cur = o1_next, o2_next

    return price_dict, occ_dict


def cluster_scenarios(
    price_dict,
    occ_dict,
    horizon,
    scenarios_to_generate,
    n_clusters,
    csv_timeseries_path=None,
    csv_probabilities_path=None,
):
    feature_matrix = np.array([
        [price_dict[t, s] for t in range(horizon)]
        + [occ_dict[1, t, s] for t in range(horizon)]
        + [occ_dict[2, t, s] for t in range(horizon)]
        for s in range(scenarios_to_generate)
    ])

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(scaled_matrix)
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    probabilities = cluster_sizes / scenarios_to_generate

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    clustered_rows = []
    price_dict_clus = {}
    occ_dict_clus = {}

    for s in range(n_clusters):
        for t in range(horizon):
            price_value = float(centroids[s, t])
            occ1_value = float(max(0.0, centroids[s, horizon + t]))
            occ2_value = float(max(0.0, centroids[s, 2 * horizon + t]))

            price_dict_clus[t, s] = price_value
            occ_dict_clus[1, t, s] = occ1_value
            occ_dict_clus[2, t, s] = occ2_value

            clustered_rows.append({
                "cluster_id": s,
                "hour": t,
                "probability": float(probabilities[s]),
                "price": price_value,
                "occ1": occ1_value,
                "occ2": occ2_value,
            })

    if csv_timeseries_path is not None:
        pd.DataFrame(clustered_rows).to_csv(csv_timeseries_path, index=False)

    if csv_probabilities_path is not None:
        pd.DataFrame(
            {
                "cluster_id": list(range(n_clusters)),
                "probability": [float(probabilities[s]) for s in range(n_clusters)],
            }
        ).to_csv(csv_probabilities_path, index=False)

    return price_dict_clus, occ_dict_clus, probabilities


def main():
    np.random.seed(RANDOM_SEED)

    data = get_fixed_data()
    hours_per_day = int(data["num_timeslots"])
    n_days = 100
    horizon = n_days * hours_per_day

    price_now = float(data["price_t"])
    price_prev = float(data["price_previous"])
    occ_r1_now = float(data["Occ1"])
    occ_r2_now = float(data["Occ2"])

    price_dict, occ_dict = generate_scenarios(
        price_now=price_now,
        price_prev=price_prev,
        occ_r1_now=occ_r1_now,
        occ_r2_now=occ_r2_now,
        horizon=horizon,
        n_scenarios=RAW_SCENARIOS,
    )

    price_dict_clus, occ_dict_clus, probabilities = cluster_scenarios(
        price_dict=price_dict,
        occ_dict=occ_dict,
        horizon=horizon,
        scenarios_to_generate=RAW_SCENARIOS,
        n_clusters=N_CENTROIDS,
        csv_timeseries_path=Path(__file__).resolve().parent / "Data" / "clustered_scenarios_timeseries.csv",
        csv_probabilities_path=Path(__file__).resolve().parent / "Data" / "clustered_scenarios_probabilities.csv",
    )

    print(
        f"Saved clustered timeseries to: "
        f"{Path(__file__).resolve().parent / 'Data' / 'clustered_scenarios_timeseries.csv'}"
    )
    print(
        f"Saved cluster probabilities to: "
        f"{Path(__file__).resolve().parent / 'Data' / 'clustered_scenarios_probabilities.csv'}"
    )
    print(f"Raw scenarios generated: {RAW_SCENARIOS}")
    print(f"Centroids: {N_CENTROIDS}")
    print(f"Cluster probabilities: {probabilities.tolist()}")
    print(f"Price dict entries: {len(price_dict_clus)}")
    print(f"Occupancy dict entries: {len(occ_dict_clus)}")


if __name__ == "__main__":
    main()

