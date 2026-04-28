import pickle
import pandas as pd

with open("stk_80_lumi.pickle", "rb") as f:
    sim = pickle.load(f)

df = sim.steps_info_to_dataframe()

print(df.tail(10).to_string())

print("\nColumns:")
print(df.columns)

print("\nMean last 50 steps:")
print(df.tail(50).mean(numeric_only=True).sort_values(ascending=False))

print("\nTotal time by part:")
part_cols = [c for c in df.columns if c not in ["simulation_offset", "step", "n_active", "duration"]]
print(df[part_cols].sum().sort_values(ascending=False))