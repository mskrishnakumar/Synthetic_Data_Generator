import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Create real dataset
real_data = pd.DataFrame({
    "instrument_type": np.random.choice(["Bond", "Loan", "Derivative"], 100),
    "market_price_available": np.random.choice([True, False], 100),
    "pricing_input_type": np.random.choice(["Observable", "Unobservable"], 100),
    "valuation_method": np.random.choice(["Mark-to-Market", "Model-based"], 100),
    "trade_volume": np.random.randint(1000, 100000, 100),
    "trade_currency": np.random.choice(["INR", "USD", "EUR"], 100),
    "instrument_rating": np.random.choice(["AAA", "AA", "A", "BBB", "BB", "B"], 100),
    "price_volatility_30d": np.round(np.random.uniform(0.1, 1.0, 100), 2),
    "days_since_last_trade": np.random.randint(0, 365, 100),
    "avg_curve_observability": np.round(np.random.uniform(0.1, 1.0, 100), 2)
})

# Detect metadata (safe even with warning)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_data)

# Train SDV model
model = GaussianCopulaSynthesizer(metadata)
model.fit(real_data)

# Generate synthetic data
synthetic_data = model.sample(50)

# Add IFRS13 level
def assign_ifrs13_level(row):
    if row['market_price_available'] and row['pricing_input_type'] == 'Observable' and row['avg_curve_observability'] > 0.85:
        return 'Level 1'
    elif row['pricing_input_type'] == 'Observable':
        return 'Level 2'
    else:
        return 'Level 3'

synthetic_data["ifrs13_level"] = synthetic_data.apply(assign_ifrs13_level, axis=1)

# Save output to csv
synthetic_data.to_csv("synthetic_ifrs13_data_sdv.csv", index=False)
print("✅ File written: synthetic_ifrs13_data_sdv.csv")

# print few rows of the synthetic data
print(synthetic_data.head())