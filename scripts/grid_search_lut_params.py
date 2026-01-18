import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# LUT Configuration
LUT_SIZE = 256

# Grid search ranges (powers of 2 only)
INPUT_DIVISORS = [32, 64, 128, 256, 512]
OUTPUT_SCALES = [4096, 8192, 16384, 32768, 65536]

def build_lut(input_divisor, output_scale):
    """Build a LUT with given parameters"""
    lut = []

    for i in range(LUT_SIZE):
        input_val = i * input_divisor

        if i == 0 or input_val < 1e-9:
            lut_value = 255
        else:
            inv_sqrt = 1.0 / np.sqrt(input_val)
            scaled = inv_sqrt * output_scale
            lut_value = int(np.round(scaled))
            lut_value = max(0, min(255, lut_value))

        lut.append(lut_value)

    return np.array(lut)

def inv_sqrt_lut(var_hw, lut, input_divisor, output_scale):
    """Simulate LUT lookup"""
    index = int(var_hw / input_divisor)
    if index < 0:
        index = 0
    if index >= LUT_SIZE:
        index = LUT_SIZE - 1

    lut_value = lut[index]
    return float(lut_value) / output_scale

def calculate_mse(var_hw_array, input_divisor, output_scale):
    """Calculate MSE for given parameters"""
    # Build LUT
    lut = build_lut(input_divisor, output_scale)

    # Calculate true values
    true_values = 1.0 / np.sqrt(var_hw_array)

    # Calculate LUT predictions
    lut_predictions = np.array([
        inv_sqrt_lut(var_hw, lut, input_divisor, output_scale)
        for var_hw in var_hw_array
    ])

    # Calculate MSE
    mse = np.mean((lut_predictions - true_values) ** 2)

    # Also calculate other metrics
    max_error = np.max(np.abs(lut_predictions - true_values))
    mean_rel_error = np.mean(np.abs(lut_predictions - true_values) / true_values) * 100

    # Check for overflow
    max_lut_value = np.max(lut)
    overflow = (max_lut_value >= 255)

    return mse, max_error, mean_rel_error, overflow

def main():
    print("="*80)
    print("LUT PARAMETER GRID SEARCH")
    print("="*80)

    # Load var_hw values
    var_hw_list = []
    with open("data/lut/var_hw_analysis.txt", 'r') as f:
        lines = f.readlines()
        reading_data = False
        for line in lines:
            if line.startswith("All var_hw values:"):
                reading_data = True
                continue
            if reading_data and ',' in line:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    var_hw_list.append(float(parts[1]))

    var_hw_array = np.array(var_hw_list)
    print(f"\nLoaded {len(var_hw_array)} var_hw values")
    print(f"Range: [{np.min(var_hw_array):.2f}, {np.max(var_hw_array):.2f}]")

    # Grid search
    print(f"\nTesting {len(INPUT_DIVISORS)} × {len(OUTPUT_SCALES)} = {len(INPUT_DIVISORS) * len(OUTPUT_SCALES)} combinations...")

    results = []
    mse_matrix = np.zeros((len(OUTPUT_SCALES), len(INPUT_DIVISORS)))

    for i, output_scale in enumerate(OUTPUT_SCALES):
        for j, input_divisor in enumerate(INPUT_DIVISORS):
            mse, max_error, mean_rel_error, overflow = calculate_mse(
                var_hw_array, input_divisor, output_scale
            )

            mse_matrix[i, j] = mse

            results.append({
                'input_divisor': input_divisor,
                'output_scale': output_scale,
                'mse': mse,
                'max_error': max_error,
                'mean_rel_error': mean_rel_error,
                'overflow': overflow
            })

    # Sort by MSE
    results.sort(key=lambda x: x['mse'])

    # Print top 5
    print("\n" + "="*80)
    print("TOP 5 BEST COMBINATIONS (Lowest MSE)")
    print("="*80)
    print(f"{'Rank':<6} {'INPUT_DIV':>12} {'OUTPUT_SCALE':>14} {'MSE':>15} "
          f"{'Max Error':>12} {'Mean Rel%':>12} {'Overflow':>10}")
    print("-"*80)

    for rank, result in enumerate(results[:5], 1):
        print(f"{rank:<6} {result['input_divisor']:>12} {result['output_scale']:>14} "
              f"{result['mse']:>15.2e} {result['max_error']:>12.6f} "
              f"{result['mean_rel_error']:>11.4f}% {'YES' if result['overflow'] else 'NO':>10}")

    # Print worst 3 for comparison
    print("\n" + "="*80)
    print("WORST 3 COMBINATIONS (Highest MSE)")
    print("="*80)
    print(f"{'Rank':<6} {'INPUT_DIV':>12} {'OUTPUT_SCALE':>14} {'MSE':>15} "
          f"{'Max Error':>12} {'Mean Rel%':>12} {'Overflow':>10}")
    print("-"*80)

    for rank, result in enumerate(results[-3:], len(results)-2):
        print(f"{rank:<6} {result['input_divisor']:>12} {result['output_scale']:>14} "
              f"{result['mse']:>15.2e} {result['max_error']:>12.6f} "
              f"{result['mean_rel_error']:>11.4f}% {'YES' if result['overflow'] else 'NO':>10}")

    # Create heatmap
    print("\n" + "="*80)
    print("GENERATING HEATMAP...")
    print("="*80)

    plt.figure(figsize=(12, 8))

    # Convert MSE to log scale for better visualization
    mse_matrix_log = np.log10(mse_matrix)

    # Create heatmap
    ax = sns.heatmap(
        mse_matrix_log,
        annot=mse_matrix,  # Show actual MSE values
        fmt='.2e',  # Scientific notation
        cmap='RdYlGn_r',  # Red (bad) to Green (good), reversed
        xticklabels=[f'{d}' for d in INPUT_DIVISORS],
        yticklabels=[f'{s}' for s in OUTPUT_SCALES],
        cbar_kws={'label': 'log10(MSE)'},
        linewidths=0.5,
        linecolor='black'
    )

    plt.title('LUT Parameter Grid Search - MSE Heatmap\n(Lower is Better)', fontsize=16, fontweight='bold')
    plt.xlabel('INPUT_DIVISOR', fontsize=14, fontweight='bold')
    plt.ylabel('OUTPUT_SCALE', fontsize=14, fontweight='bold')

    # Add text annotation for best combination
    best = results[0]
    best_i = OUTPUT_SCALES.index(best['output_scale'])
    best_j = INPUT_DIVISORS.index(best['input_divisor'])

    # Add a star to mark the best
    ax.text(best_j + 0.5, best_i + 0.5, '★',
            ha='center', va='center', fontsize=30, color='blue', fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = "data/lut"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "lut_grid_search_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_file}")

    # Create second heatmap for relative error
    plt.figure(figsize=(12, 8))

    # Extract relative errors
    rel_error_matrix = np.zeros((len(OUTPUT_SCALES), len(INPUT_DIVISORS)))
    for result in results:
        i = OUTPUT_SCALES.index(result['output_scale'])
        j = INPUT_DIVISORS.index(result['input_divisor'])
        rel_error_matrix[i, j] = result['mean_rel_error']

    ax = sns.heatmap(
        rel_error_matrix,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',
        xticklabels=[f'{d}' for d in INPUT_DIVISORS],
        yticklabels=[f'{s}' for s in OUTPUT_SCALES],
        cbar_kws={'label': 'Mean Relative Error (%)'},
        linewidths=0.5,
        linecolor='black'
    )

    plt.title('LUT Parameter Grid Search - Mean Relative Error %\n(Lower is Better)', fontsize=16, fontweight='bold')
    plt.xlabel('INPUT_DIVISOR', fontsize=14, fontweight='bold')
    plt.ylabel('OUTPUT_SCALE', fontsize=14, fontweight='bold')

    # Mark best
    ax.text(best_j + 0.5, best_i + 0.5, '★',
            ha='center', va='center', fontsize=30, color='blue', fontweight='bold')

    plt.tight_layout()

    output_file2 = os.path.join(output_dir, "lut_grid_search_rel_error.png")
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Relative error heatmap saved to: {output_file2}")

    # Save detailed results to file
    results_file = os.path.join(output_dir, "grid_search_results.txt")
    with open(results_file, 'w') as f:
        f.write("LUT Parameter Grid Search Results\n")
        f.write("="*80 + "\n\n")

        f.write("TOP 5 BEST COMBINATIONS:\n")
        f.write("-"*80 + "\n")
        for rank, result in enumerate(results[:5], 1):
            f.write(f"\nRank {rank}:\n")
            f.write(f"  INPUT_DIVISOR:  {result['input_divisor']} (2^{int(np.log2(result['input_divisor']))})\n")
            f.write(f"  OUTPUT_SCALE:   {result['output_scale']} (2^{int(np.log2(result['output_scale']))})\n")
            f.write(f"  MSE:            {result['mse']:.10e}\n")
            f.write(f"  Max Error:      {result['max_error']:.10f}\n")
            f.write(f"  Mean Rel Error: {result['mean_rel_error']:.6f}%\n")
            f.write(f"  Overflow:       {'YES' if result['overflow'] else 'NO'}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("ALL RESULTS (sorted by MSE):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'INPUT_DIV':>12} {'OUTPUT_SCALE':>14} {'MSE':>15} {'Max Error':>12} {'Mean Rel%':>12} {'Overflow':>10}\n")
        f.write("-"*80 + "\n")
        for result in results:
            f.write(f"{result['input_divisor']:>12} {result['output_scale']:>14} "
                   f"{result['mse']:>15.2e} {result['max_error']:>12.6f} "
                   f"{result['mean_rel_error']:>11.4f}% {'YES' if result['overflow'] else 'NO':>10}\n")

    print(f"Detailed results saved to: {results_file}")

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE!")
    print("="*80)
    print(f"\nBest configuration:")
    print(f"  INPUT_DIVISOR = {results[0]['input_divisor']} (2^{int(np.log2(results[0]['input_divisor']))})")
    print(f"  OUTPUT_SCALE = {results[0]['output_scale']} (2^{int(np.log2(results[0]['output_scale']))})")
    print(f"  MSE = {results[0]['mse']:.2e}")
    print(f"  Mean Relative Error = {results[0]['mean_rel_error']:.4f}%")

    if results[0]['input_divisor'] == 128 and results[0]['output_scale'] == 16384:
        print("\n  ✓ This matches our current configuration!")
    else:
        print("\n  ⚠ This differs from current configuration (128, 16384)")
        print(f"  Consider updating to the optimal values above.")

if __name__ == "__main__":
    main()
