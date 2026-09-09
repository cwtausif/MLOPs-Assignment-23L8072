print("Loading data...")
X = data[features]
X = (data[features] - data[features].mean()) / data[features].std()