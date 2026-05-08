import pandas as pd
from pathlib import Path
import numpy as np


data_dir = Path(__file__).resolve().parent / "Data"

occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv").values.flatten()
occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv").values.flatten()

df_prices = pd.read_csv(data_dir / "v2_PriceData.csv") # skip header, as we will access the columns by index
