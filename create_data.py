import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_dates(start='2020-01-01', periods=1000):
    return pd.date_range(start=start, periods=periods, freq='D')

def generate_real_and_synthetic_data(n_samples=1000, output_dir="data", with_outliers=True):
    os.makedirs(output_dir, exist_ok=True)

    # Generate Real Data
    age = np.random.normal(35, 10, n_samples).astype(int)
    income = np.random.normal(70000, 20000, n_samples)
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples)
    region = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    signup_date = generate_dates('2020-01-01', n_samples)
    score = np.clip(np.random.normal(0.6, 0.1, n_samples), 0, 1)
    num_logins = np.random.poisson(30, n_samples)
    is_active = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    purchased = (score + (income > 80000).astype(int) + is_active) > 1.5
    purchased = purchased.astype(int)
    time_spent = score * 50 + np.random.normal(0, 5, n_samples)
    num_clicks = time_spent * 1.5 + np.random.normal(0, 10, n_samples)

    real_df = pd.DataFrame({
        'user_id': np.arange(1, n_samples + 1),
        'age': age,
        'income': income,
        'gender': gender,
        'region': region,
        'signup_date': signup_date,
        'score': score,
        'num_logins': num_logins,
        'is_active': is_active,
        'purchased': purchased,
        'time_spent': time_spent,
        'num_clicks': num_clicks
    })

    # Generate Synthetic Data
    synth_df = real_df.copy()

    # Inject noise and outliers
    synth_df['age'] = (synth_df['age'] + np.random.normal(0, 5, n_samples)).astype(int)
    synth_df['income'] = synth_df['income'] * np.random.normal(1.0, 0.1, n_samples)

    # Add outliers if enabled
    if with_outliers:
        outlier_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        synth_df.loc[outlier_idx, 'income'] *= np.random.choice([5, 10], size=len(outlier_idx))  # extreme income
        synth_df.loc[outlier_idx, 'age'] += np.random.choice([40, 60], size=len(outlier_idx))    # extreme age
        synth_df.loc[outlier_idx, 'score'] = np.random.uniform(0, 1, size=len(outlier_idx))      # completely random scores

    synth_df['score'] = np.clip(synth_df['score'] + np.random.normal(0, 0.08, n_samples), 0, 1)
    synth_df['num_logins'] += np.random.poisson(3, n_samples)
    synth_df['time_spent'] = synth_df['score'] * 45 + np.random.normal(0, 10, n_samples)
    synth_df['num_clicks'] = synth_df['time_spent'] * 1.3 + np.random.normal(0, 15, n_samples)

    # Slight change in target logic
    synth_df['is_active'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    synth_df['purchased'] = ((synth_df['score'] + (synth_df['income'] > 90000).astype(int) + synth_df['is_active']) > 1.8).astype(int)

    # Save to CSV
    real_df.to_csv(os.path.join(output_dir, "real_data.csv"), index=False)
    synth_df.to_csv(os.path.join(output_dir, "synthetic_data.csv"), index=False)

    print("✅ Real and synthetic data saved with outliers" if with_outliers else "✅ Synthetic data without outliers")

# Example usage
generate_real_and_synthetic_data(n_samples=1000, output_dir="data", with_outliers=False)