print("Loading data...")
X = data[features]
X = (data[features] - data[features].min()) / (data[features].max() - data[features].min())