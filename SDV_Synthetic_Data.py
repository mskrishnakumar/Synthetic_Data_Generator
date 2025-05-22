import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Set seed
np.random.seed(42)

# Helper: generate random past date
def random_past_date(start_days_ago=730, end_days_ago=1):
    days_ago = np.random.randint(end_days_ago, start_days_ago)
    return datetime.today().date() - timedelta(days=days_ago)

# Master lists and mappings
instrument_types = ["Bond", "Traded Equity", "Interest Rate Derivative"]
ir_subtypes = ["IRS", "Basis Swap", "Cap", "Floor", "Swaption", "CMS"]
curve_map = {
    "Bond": ["Govt Curve", "Corporate Curve"],
    "Traded Equity": ["Equity Index Curve", "Spot Curve"],
    "Interest Rate Derivative": ["Swap Curve", "Forward Rate Curve"]
}
ref_rate_map = {
    "USD": ["USD_LIBOR_6M", "SOFR", "SONIA"],
    "EUR": ["EURIBOR", "ESTR"],
    "INR": ["MIFOR", "T-Bill_91D"]
}
curve_terms = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"]
real_curves = {
    "SONIA":        [0.15, 0.25, 0.45, 0.75, 1.2, 2.3, 3.5],
    "USD_LIBOR_6M": [0.5, 1.1, 1.8, 2.4, 3.1, 4.6, 5.3],
    "EURIBOR":      [-0.3, 0.0, 0.4, 0.8, 1.6, 2.9, 3.7],
    "MIFOR":        [3.7, 4.5, 5.2, 5.9, 6.4, 7.2, 7.8],
    "SOFR":         [0.2, 0.3, 0.6, 1.0, 1.8, 3.0, 4.2],
    "ESTR":         [0.1, 0.2, 0.5, 0.9, 1.7, 2.8, 3.4],
    "T-Bill_91D":   [3.5, 3.6, 3.7, 3.8, 4.0, 4.3, 4.5]
}

# Generate base data
n = 300
df = pd.DataFrame({
    "instrument_type": np.random.choice(instrument_types, n),
    "market_price_available": np.random.choice([True, False], n),
    "pricing_input_type": np.random.choice(["Observable", "Unobservable"], n),
    "valuation_method": np.random.choice(["Mark-to-Market", "Model-based"], n),
    "trade_volume": np.random.randint(1000, 100000, n),
    "trade_currency": np.random.choice(["INR", "USD", "EUR"], n),
    "instrument_rating": np.random.choice(["AAA", "AA", "A", "BBB", "BB", "B"], n),
    "price_volatility_30d": np.round(np.random.uniform(0.1, 1.0, n), 2),
    "days_since_last_trade": np.random.randint(0, 365, n),
    "avg_curve_observability": np.round(np.random.uniform(0.1, 1.0, n), 2),
    "correlation_risk": np.random.choice([True, False], n)
})

# Assign derived features (unchanged)
df["curve_used_for_pricing"] = df["instrument_type"].apply(lambda x: np.random.choice(curve_map[x]))
df["curve_observability"] = pd.cut(df["avg_curve_observability"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
df["trade_date"] = pd.to_datetime([random_past_date() for _ in range(n)])
df["maturity_date"] = df["trade_date"] + pd.to_timedelta(np.random.randint(30, 3650, n), unit="D")
df["tenor_days"] = (df["maturity_date"] - df["trade_date"]).dt.days
df["reference_rate_curve"] = df["trade_currency"].apply(lambda c: np.random.choice(ref_rate_map[c]))
df["reference_rate_value"] = df["reference_rate_curve"].apply(lambda r: round(np.random.uniform(0.1, 8.0), 2))

# IR Derivatives
df["ir_derivative_type"] = df.apply(
    lambda row: np.random.choice(ir_subtypes) if row["instrument_type"] == "Interest Rate Derivative" else None,
    axis=1
)
df["has_embedded_option"] = df["ir_derivative_type"].isin(["Cap", "Floor", "Swaption"])
df["notional_amount"] = np.where(df["instrument_type"] == "Interest Rate Derivative",
                                 np.random.randint(1_000_000, 100_000_000, n),
                                 np.nan)
df["strike_rate"] = np.where(df["has_embedded_option"], np.round(np.random.uniform(1.0, 6.0, n), 2), np.nan)
df["option_premium"] = np.where(df["has_embedded_option"], np.round(np.random.uniform(10_000, 500_000, n), 2), np.nan)

# Yield curve term and rate
df["curve_term"] = np.random.choice(curve_terms, size=n)
def get_curve_rate(row):
    curve = row["reference_rate_curve"]
    term = row["curve_term"]
    try:
        term_index = curve_terms.index(term)
        return real_curves.get(curve, [np.nan]*7)[term_index]
    except:
        return np.nan
df["curve_rate"] = df.apply(get_curve_rate, axis=1)

# ✅ BOOST Level 1-Like Examples
level1_like = df[
    (df["market_price_available"]) &
    (df["pricing_input_type"] == "Observable") &
    (df["avg_curve_observability"] > 0.85)
]

if len(level1_like) < 30:
    df = pd.concat([df, level1_like.sample(50, replace=True)], ignore_index=True)
    print(f"🔁 Added {50} more Level 1-like samples to boost distribution.")

# Metadata (unchanged)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.update_column("trade_date", sdtype="datetime")
metadata.update_column("maturity_date", sdtype="datetime")
metadata.update_column("has_embedded_option", sdtype="boolean")
metadata.update_column("market_price_available", sdtype="boolean")
metadata.update_column("correlation_risk", sdtype="boolean")
metadata.update_column("curve_observability", sdtype="categorical")
metadata.update_column("curve_used_for_pricing", sdtype="categorical")
metadata.update_column("reference_rate_curve", sdtype="categorical")
metadata.update_column("ir_derivative_type", sdtype="categorical")
metadata.update_column("curve_term", sdtype="categorical")

# Fit model and sample
synth = GaussianCopulaSynthesizer(metadata)
synth.fit(df)
synth_data = synth.sample(100)
synth_data["trade_date"] = pd.to_datetime(synth_data["trade_date"])
synth_data["maturity_date"] = pd.to_datetime(synth_data["maturity_date"])
synth_data["tenor_days"] = (synth_data["maturity_date"] - synth_data["trade_date"]).dt.days

# IFRS 13 logic
def assign_ifrs13_level(row):
    if (
        row['market_price_available']
        and row['pricing_input_type'] == 'Observable'
        and row['curve_observability'] == 'High'
        and row['avg_curve_observability'] > 0.85
    ):
        return 'Level 1'
    elif row['pricing_input_type'] == 'Observable':
        return 'Level 2'
    else:
        return 'Level 3'

synth_data["ifrs13_level"] = synth_data.apply(assign_ifrs13_level, axis=1)

# Add trade ID
synth_data.insert(0, "trade_id", ["HACK" + str(i).zfill(7) for i in range(1, len(synth_data) + 1)])

# Save
synth_data.to_csv("synthetic_ir_derivatives_with_real_curves.csv", index=False)
print(" File written: synthetic_ir_derivatives_with_real_curves.csv")

# 📊 IFRS 13 Distribution
print("\n IFRS 13 Level Distribution:\n", synth_data["ifrs13_level"].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

# Preview
print(synth_data[[
    "instrument_type", "ir_derivative_type", "reference_rate_curve",
    "curve_term", "curve_rate", "strike_rate", "option_premium", "ifrs13_level"
]].head())
print("\n IFRS 13 Level Distribution:\n", synth_data["ifrs13_level"].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
print("\n Data generation and synthesis complete.")