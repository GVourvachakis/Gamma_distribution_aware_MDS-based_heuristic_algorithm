import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from tqdm import tqdm
from time import time
from typing import Tuple

# import sci-kit learn's dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.manifold import MDS

# Load Boston Housing Dataset as a toy data set
# If you have scikit-learn 1.0+, use fetch_openml for Boston dataset.
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

def validate_sampling_distribution(y_full, y_sample, title="Distribution Comparison", dataset = "California Housing"):
    
    """Compare distributions of full and sampled data"""

    plt.figure(figsize=(10, 6))
    plt.hist(y_full, bins=50, alpha=0.5, label='Full Dataset', density=True)
    plt.hist(y_sample, bins=50, alpha=0.5, label='Sampled Dataset', density=True)
    plt.title(title)
    if dataset == "California Housing":
        plt.xlabel('House Price')
    elif dataset == "Boston Housing": 
        plt.xlabel('Median value of owner-occupied homes')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def perform_baseline_regression(X_train, y_train, X_test, y_test):
    """Perform baseline regression without dimensionality reduction"""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

def uniform_sampling_mds_analysis(X_train, y_train, X_test, y_test, sample_size=5_000, n_runs: int = 3):
    """
    Perform MDS analysis using stratified sampling with validation metrics.
    """
    results = {
        'sampling_validation': {},
        'performance_metrics': {'2D': {}, '3D': {}},
        'timing_metrics': {},
        'baseline_metrics': {}
    }

    # Start timing
    start_time = time()

    # Create bins for stratification
    n_bins = 5
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbd.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Store results for multiple runs
    results_2d = {'mse': [], 'mae': [], 'r2': []}
    results_3d = {'mse': [], 'mae': [], 'r2': []}

    # Initialize profiler
    profiler = Profiler()

    # Perform multiple runs
    n_runs = n_runs

    for run in tqdm(range(n_runs), desc="Overall runs"):
        # Stratified sampling
        train_indices = []
        samples_per_bin = sample_size // n_bins

        for bin_idx in tqdm(range(n_bins), desc="Stratified sampling", leave=False):
            bin_indices = np.where(y_binned == bin_idx)[0]
            selected_indices = np.random.choice(
                bin_indices,
                size=min(samples_per_bin, len(bin_indices)),
                replace=False
            )
            train_indices.extend(selected_indices)

        # Sample the data
        X_train_sample = X_train[train_indices]
        y_train_sample = y_train[train_indices]

        # Sample test set
        test_size = len(y_test) // 4
        test_indices = np.random.choice(len(y_test), size=test_size, replace=False)
        X_test_sample = X_test[test_indices]
        y_test_sample = y_test[test_indices]

        # Store sampling validation data for first run
        if run == 0:
            results['sampling_validation'] = {
                'y_full': y_train,
                'y_sample': y_train_sample,
                'sample_size': len(y_train_sample)
            }

        # Profile MDS transformation for both 2D and 3D
        if profiler.is_running:
            profiler.stop() # maintaining asynchronus support
        profiler.start() # Start profiling

        for n_components in [2, 3]:
            dim_start_time = time()
            tqdm.write(f"Run {run + 1}/{n_runs}: Processing {n_components}D reduction...")

            # MDS transformation
            mds = MDS(
                n_components=n_components,
                random_state=42 + run,
                n_init=1,
                max_iter=300,
                n_jobs=-1,
                dissimilarity='euclidean',
                eps=1e-3
            )

            # Transform data
            X_train_mds = mds.fit_transform(X_train_sample)
            X_test_mds = mds.fit_transform(X_test_sample)

            # Fit regression and predict
            lr = LinearRegression()
            lr.fit(X_train_mds, y_train_sample)
            y_pred = lr.predict(X_test_mds)

            # Calculate metrics
            mse = mean_squared_error(y_test_sample, y_pred)
            mae = mean_absolute_error(y_test_sample, y_pred)
            r2 = r2_score(y_test_sample, y_pred)

            # Store results
            metrics_dict = results_2d if n_components == 2 else results_3d
            metrics_dict['mse'].append(mse)
            metrics_dict['mae'].append(mae)
            metrics_dict['r2'].append(r2)

            # Store timing for first run
            if run == 0:
                results['timing_metrics'][f'{n_components}D'] = time() - dim_start_time

        profiler.stop()  # Stop profiling

    if n_runs != 0:
        # Print profiling results
        profiler.print()

    # Calculate baseline metrics
    print("\nCalculating baseline metrics...")
    results['baseline_metrics'] = perform_baseline_regression(
        X_train, y_train, X_test, y_test
    )

    # Store final performance metrics
    for dim, res in [('2D', results_2d), ('3D', results_3d)]:
        results['performance_metrics'][dim] = {
            'mse_mean': np.mean(res['mse']),
            'mse_std': np.std(res['mse']),
            'mae_mean': np.mean(res['mae']),
            'mae_std': np.std(res['mae']),
            'r2_mean': np.mean(res['r2']),
            'r2_std': np.std(res['r2'])
        }

    # Store total execution time
    results['timing_metrics']['total'] = time() - start_time

    return results

def boston_loader_scaled(microstate = 42, test_size_percentile = 0.2) -> \
                                 Tuple[Tuple[np.matrix,np.ndarray], Tuple[np.matrix,np.ndarray]]:

    data = fetch_openml(name="boston", version=1)

    X, y = data.data.to_numpy(), data.target.to_numpy()

    # Split the diabetes dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentile, random_state=microstate)

    # Initialize StandardScaler for feature scaling
    scaler = StandardScaler()

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    

    return (X_train_scaled, y_train), (X_test_scaled, y_test)

def main() -> None:
    '''
    test on smaller dataset [Boston Housing Dataset]

    Description: Contains data on housing prices in Boston suburbs.
    Size: 506 samples, 13 features
    Target: Median value of owner-occupied homes (continuous variable)
    '''

    # #Run the complete analysis
    # warnings.filterwarnings('ignore')
    # np.random.seed(42) 

    (X_train_scaled, y_train), (X_test_scaled, y_test) = boston_loader_scaled()

    # Perform analysis
    results = uniform_sampling_mds_analysis(X_train_scaled, y_train, X_test_scaled, y_test, 
                                            sample_size=5_000, n_runs=1)

    # Validate sampling distribution
    validate_sampling_distribution(
                                    results['sampling_validation']['y_full'],
                                    results['sampling_validation']['y_sample'],
                                    "Target Variable Distribution: Full vs Sampled Dataset",
                                    "Boston Housing"
                                )

    # Print comprehensive results
    print("\nAnalysis Results:")
    print("\nBaseline Metrics (No dimensionality reduction):")
    print(f"MSE: {results['baseline_metrics']['mse']:.4f}")
    print(f"MAE: {results['baseline_metrics']['mae']:.4f}")
    print(f"R²: {results['baseline_metrics']['r2']:.4f}")

    for dim in ['2D', '3D']:
        print(f"\n{dim} MDS Reduction Metrics:")
        metrics = results['performance_metrics'][dim]
        print(f"MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
        print(f"MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
        print(f"R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")

    print("\nExecution Times:")
    for key, value in results['timing_metrics'].items():
        print(f"{key}: {value:.2f} seconds")


if __name__ == "__main__":
    main()


