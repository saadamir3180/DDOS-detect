"""
Create sample traffic data files for testing predictions.
This extracts samples from the main dataset without labels.
"""

import pandas as pd

print("Creating sample traffic data files...")

# Load the dataset
df = pd.read_csv("data/raw/cicddos2019_dataset.csv")

# Remove label columns
feature_columns = [col for col in df.columns if col not in ['Label', 'Class']]
df_features = df[feature_columns]

# Create different sample files
print("\n1. Creating normal traffic sample (100 rows)...")
normal_sample = df[df['Label'] == 'Benign'][feature_columns].head(100)
normal_sample.to_csv("data/samples/normal_traffic.csv", index=False)
print(f"   Saved: data/samples/normal_traffic.csv ({len(normal_sample)} rows)")

print("\n2. Creating attack traffic sample (100 rows)...")
attack_sample = df[df['Label'] != 'Benign'][feature_columns].head(100)
attack_sample.to_csv("data/samples/attack_traffic.csv", index=False)
print(f"   Saved: data/samples/attack_traffic.csv ({len(attack_sample)} rows)")

print("\n3. Creating mixed traffic sample (200 rows)...")
mixed_sample = df[feature_columns].sample(n=200, random_state=42)
mixed_sample.to_csv("data/samples/mixed_traffic.csv", index=False)
print(f"   Saved: data/samples/mixed_traffic.csv ({len(mixed_sample)} rows)")

print("\n4. Creating test traffic with labels (for verification)...")
test_with_labels = df.sample(n=100, random_state=42)
test_with_labels.to_csv("data/samples/test_with_labels.csv", index=False)
print(f"   Saved: data/samples/test_with_labels.csv ({len(test_with_labels)} rows)")

print("\n" + "="*60)
print("✅ Sample files created successfully!")
print("="*60)
print("\nYou can now use these files in the GUI:")
print("  - normal_traffic.csv - Only benign traffic")
print("  - attack_traffic.csv - Only attack traffic")
print("  - mixed_traffic.csv - Mix of both")
print("  - test_with_labels.csv - For comparing predictions")
print("\nFeatures in each file:", len(feature_columns))
print("\nTo use in GUI:")
print("  1. Go to 'Single Prediction' or 'Batch Prediction' tab")
print("  2. Load your trained model")
print("  3. Select one of these CSV files")
print("  4. Click Predict!")
