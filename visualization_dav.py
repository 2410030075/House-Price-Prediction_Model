"""
House Price Prediction - Data Visualization Script
Complete visualization suite for exploratory data analysis and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style and color palette
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create plots directory
Path('plots').mkdir(exist_ok=True)

# ========================================
# Load Dataset
# ========================================
# Replace with your actual dataset file
df = pd.read_csv('Melbourne_imputed.csv')

# Ensure required columns exist
required_columns = ['Price', 'Landsize', 'Bedroom2', 'Bathroom', 'Regionname']
# Map to standardized names if needed
column_mapping = {
    'Price': 'price',
    'Landsize': 'area',
    'Bedroom2': 'bedrooms',
    'Bathroom': 'bathrooms',
    'Regionname': 'location'
}

# Create working dataframe with standardized column names
df_viz = df.copy()
for old_col, new_col in column_mapping.items():
    if old_col in df_viz.columns:
        df_viz[new_col] = df_viz[old_col]

# ========================================
# 1. Price Distribution
# ========================================
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df_viz['price'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df_viz['price'].dropna()), bins=50, color='coral', edgecolor='black', alpha=0.7)
plt.title('Distribution of House Prices (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Log(Price)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/price_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/price_distribution.png")
plt.close()

# ========================================
# 2. Scatter Plot (Area vs Price)
# ========================================
plt.figure(figsize=(10, 6))
plt.scatter(df_viz['area'].dropna(), df_viz['price'].dropna(), 
            alpha=0.5, c='teal', edgecolors='black', linewidth=0.5, s=50)
plt.title('House Price vs Area', fontsize=14, fontweight='bold')
plt.xlabel('Area (sq ft)', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
mask = df_viz['area'].notna() & df_viz['price'].notna()
z = np.polyfit(df_viz.loc[mask, 'area'], df_viz.loc[mask, 'price'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_viz['area'].min(), df_viz['area'].max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend Line')
plt.legend()

plt.tight_layout()
plt.savefig('plots/area_vs_price.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/area_vs_price.png")
plt.close()

# ========================================
# 3. Boxplot (Price by Location)
# ========================================
plt.figure(figsize=(14, 6))
location_data = df_viz[['location', 'price']].dropna()

# Limit to top locations by count for readability
top_locations = location_data['location'].value_counts().head(10).index
location_filtered = location_data[location_data['location'].isin(top_locations)]

sns.boxplot(data=location_filtered, x='location', y='price', palette='Set2')
plt.title('House Price Distribution by Location (Top 10)', fontsize=14, fontweight='bold')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/price_by_location.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/price_by_location.png")
plt.close()

# ========================================
# 4. Heatmap (Feature Correlations)
# ========================================
plt.figure(figsize=(10, 8))

# Select numeric columns for correlation
numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms']
available_cols = [col for col in numeric_cols if col in df_viz.columns]
corr_matrix = df_viz[available_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/correlation_heatmap.png")
plt.close()

# ========================================
# 5. Pairplot (Relationships)
# ========================================
print("Generating pairplot (this may take a moment)...")
# Use a sample for large datasets
sample_size = min(1000, len(df_viz))
df_sample = df_viz[available_cols].dropna().sample(n=sample_size, random_state=42)

pairplot = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                        corner=False, palette='viridis')
pairplot.fig.suptitle('Feature Relationships (Pairplot)', y=1.01, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/pairplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/pairplot.png")
plt.close()

# ========================================
# 6. Actual vs Predicted Prices
# ========================================
# Note: This requires y_test and y_pred from your model
# Example placeholder data - replace with your actual predictions

try:
    # Attempt to load predictions if they exist
    predictions_df = pd.read_csv('predictions.csv')
    y_test = predictions_df['actual'].values
    y_pred = predictions_df['predicted'].values
except:
    # Generate example data for demonstration
    print("Note: Using synthetic data for predictions. Replace with actual y_test and y_pred.")
    np.random.seed(42)
    n_samples = 200
    y_test = np.random.uniform(200000, 2000000, n_samples)
    y_pred = y_test + np.random.normal(0, 100000, n_samples)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, c='mediumseagreen', edgecolors='black', linewidth=0.5, s=50)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Add R² score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
plt.text(0.05, 0.95, f'R² Score: {r2:.3f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/actual_vs_predicted.png")
plt.close()

# ========================================
# 7. Feature Importance Plot
# ========================================
# Note: This requires feature_importances_ from tree-based model
# Example placeholder - replace with actual feature importances

try:
    # Attempt to load feature importances if they exist
    importance_df = pd.read_csv('feature_importance.csv')
    features = importance_df['feature'].values
    importances = importance_df['importance'].values
except:
    # Generate example data for demonstration
    print("Note: Using example feature importances. Replace with actual model.feature_importances_")
    features = np.array(['area', 'bedrooms', 'bathrooms', 'location_encoded', 
                        'property_age', 'distance_to_cbd', 'num_rooms'])
    importances = np.array([0.35, 0.15, 0.12, 0.20, 0.08, 0.06, 0.04])

# Sort by importance
indices = np.argsort(importances)[::-1]
features_sorted = features[indices]
importances_sorted = importances[indices]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(importances_sorted)), importances_sorted, 
                color=sns.color_palette('viridis', len(importances_sorted)), 
                edgecolor='black', alpha=0.8)
plt.yticks(range(len(features_sorted)), features_sorted)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance (Tree-Based Model)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (feat, imp) in enumerate(zip(features_sorted, importances_sorted)):
    plt.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/feature_importance.png")
plt.close()

# ========================================
# 8. Residual Plot
# ========================================
residuals = y_test - y_pred

plt.figure(figsize=(12, 5))

# Residuals vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6, c='purple', edgecolors='black', linewidth=0.5, s=50)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.grid(True, alpha=0.3)

# Residuals distribution
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, color='orchid', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/residual_plot.png")
plt.close()
# ========================================
# Advanced Visualizations (9-15)
# ========================================

# Create advanced_plots directory
Path('advanced_plots').mkdir(exist_ok=True)

# 9. Distribution of Categorical Features (countplot of bedrooms and top locations)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Countplot of bedrooms (converted to integer for categorical counts)
if 'bedrooms' in df_viz.columns:
    sns.countplot(x=df_viz['bedrooms'].dropna().astype(int), palette='pastel')
    plt.title('Count of Bedrooms', fontsize=13, fontweight='bold')
    plt.xlabel('Bedrooms', fontsize=11)
    plt.ylabel('Count', fontsize=11)
else:
    plt.text(0.5, 0.5, 'bedrooms column not available', ha='center')

plt.subplot(1, 2, 2)
if 'location' in df_viz.columns:
    top_locs = df_viz['location'].value_counts().nlargest(10).index
    sns.countplot(y=df_viz[df_viz['location'].isin(top_locs)]['location'], 
                  order=top_locs, palette='magma')
    plt.title('Top 10 Locations by Count', fontsize=13, fontweight='bold')
    plt.xlabel('Count', fontsize=11)
    plt.ylabel('Location', fontsize=11)
else:
    plt.text(0.5, 0.5, 'location column not available', ha='center')

plt.tight_layout()
plt.savefig('advanced_plots/count_categorical_features.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# What this plot shows (comment):
# - Left: distribution of number of bedrooms in the dataset (useful for categorical imbalance)
# - Right: top 10 locations by sample count (helps identify location coverage)

# 10. Training vs Test Performance Comparison (barplot of R² and MSE if train metrics available)
from sklearn.metrics import mean_squared_error
metrics = {}
metrics_labels = []
metrics_values_r2 = []
metrics_values_mse = []

# Try to detect training metrics from predictions_df if present
has_train = False
try:
    if 'train_actual' in predictions_df.columns and 'train_predicted' in predictions_df.columns:
        y_train = predictions_df['train_actual'].values
        y_train_pred = predictions_df['train_predicted'].values
        has_train = True
except NameError:
    # predictions_df not available beyond earlier scope
    has_train = False

# Compute test metrics
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
metrics_labels.append('Test')
metrics_values_r2.append(r2_test)
metrics_values_mse.append(mse_test)

if has_train:
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    metrics_labels.insert(0, 'Train')
    metrics_values_r2.insert(0, r2_train)
    metrics_values_mse.insert(0, mse_train)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=metrics_labels, y=metrics_values_r2, palette='Blues')
plt.title('R²: Training vs Test', fontsize=13, fontweight='bold')
plt.ylabel('R² Score', fontsize=11)
for i, v in enumerate(metrics_values_r2):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.subplot(1, 2, 2)
sns.barplot(x=metrics_labels, y=metrics_values_mse, palette='Reds')
plt.title('MSE: Training vs Test', fontsize=13, fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=11)
for i, v in enumerate(metrics_values_mse):
    plt.text(i, v + max(metrics_values_mse) * 0.01, f'{v:.0f}', ha='center')

plt.tight_layout()
plt.savefig('advanced_plots/train_vs_test_performance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Comment: This compares training and test performance using R² and MSE to detect overfitting/underfitting.

# 11. 3D Scatter Plot (Area, Bedrooms, Price)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
mask3d = df_viz[['area', 'bedrooms', 'price']].dropna().index
if len(mask3d) > 0:
    x = df_viz.loc[mask3d, 'area']
    y = df_viz.loc[mask3d, 'bedrooms'].astype(float)
    z = df_viz.loc[mask3d, 'price']
    p = ax.scatter(x, y, z, c=z, cmap='viridis', depthshade=True, s=20, edgecolor='k', alpha=0.7)
    ax.set_xlabel('Area', fontsize=11)
    ax.set_ylabel('Bedrooms', fontsize=11)
    ax.set_zlabel('Price', fontsize=11)
    ax.set_title('3D Scatter: Area vs Bedrooms vs Price', fontsize=13, fontweight='bold')
    fig.colorbar(p, ax=ax, shrink=0.6, label='Price')
    plt.tight_layout()
    plt.savefig('advanced_plots/3d_scatter_area_bedrooms_price.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    plt.text(0.5, 0.5, 'Insufficient data for 3D scatter', ha='center')
    plt.savefig('advanced_plots/3d_scatter_area_bedrooms_price.png', dpi=300, bbox_inches='tight')
    plt.show()
plt.close()

# Comment: 3D plot shows relationship between area, bedroom count and price; color indicates price magnitude.

# 12. Heatmap on Map (geographical price visualization if latitude & longitude exist)
try:
    import folium
    from folium.plugins import HeatMap
    folium_available = True
except Exception:
    folium_available = False

if folium_available:
    if 'latitude' in df_viz.columns and 'longitude' in df_viz.columns and 'price' in df_viz.columns:
        coords = df_viz[['latitude', 'longitude', 'price']].dropna()
        if len(coords) > 0:
            # Center map at mean location
            center = [coords['latitude'].mean(), coords['longitude'].mean()]
            m = folium.Map(location=center, zoom_start=11, tiles='cartodbpositron')
            # Prepare data for heatmap: lat, lon, weight
            heat_data = coords[['latitude', 'longitude', 'price']].values.tolist()
            # Normalize weights to a reasonable scale
            max_price = coords['price'].max()
            HeatMap([[row[0], row[1], row[2] / max_price] for row in heat_data], radius=12, blur=10,
                    max_val=1.0).add_to(m)
            m.save('advanced_plots/price_heatmap_map.html')
            print('✓ Saved: advanced_plots/price_heatmap_map.html')
        else:
            print('Latitude/Longitude present but no complete rows to generate map.')
    else:
        print('Latitude/Longitude columns not found — skipping map heatmap. Add `latitude` and `longitude` to dataset to enable.')
else:
    print('folium not installed — skipping geographic heatmap. To enable, install folium (pip install folium).')

# Comment: The folium heatmap shows geographic concentration of prices; weights represent relative price intensity.

# 13. Interactive Scatter Plot using Plotly (Area vs Price colored by location)
import plotly.express as px
if {'area', 'price', 'location'}.issubset(df_viz.columns):
    df_plotly = df_viz[['area', 'price', 'location']].dropna()
    # limit for interactivity
    df_plotly_sample = df_plotly.sample(n=min(len(df_plotly), 2000), random_state=42)
    fig = px.scatter(df_plotly_sample, x='area', y='price', color='location', hover_data=['area','price'],
                     title='Interactive Scatter: Area vs Price (colored by location)', labels={'area':'Area','price':'Price'})
    fig.write_html('advanced_plots/interactive_area_price.html')
    # also save static image if kaleido is available
    try:
        fig.write_image('advanced_plots/interactive_area_price.png', scale=2)
    except Exception:
        pass
    fig.show()
    print('✓ Saved: advanced_plots/interactive_area_price.html')
else:
    print('Missing columns for interactive plot: need area, price, location')

# Comment: Interactive Plotly scatter allows zooming and hover to inspect individual data points across locations.

# 14. Enhanced Pairplot (area, bedrooms, bathrooms, price colored by location)
pair_cols = ['area', 'bedrooms', 'bathrooms', 'price']
available = [c for c in pair_cols if c in df_viz.columns]
if 'location' in df_viz.columns and len(available) >= 2:
    # Limit categories and sample size for clarity
    top_locs = df_viz['location'].value_counts().nlargest(6).index
    df_pair = df_viz[df_viz['location'].isin(top_locs)][available + ['location']].dropna()
    sample_n = min(len(df_pair), 1000)
    df_pair_sample = df_pair.sample(sample_n, random_state=42)
    pp = sns.pairplot(df_pair_sample, vars=available, hue='location', diag_kind='kde', plot_kws={'alpha':0.6, 's':30}, palette='tab10')
    pp.fig.suptitle('Enhanced Pairplot (sampled, colored by location)', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    pp.savefig('advanced_plots/enhanced_pairplot.png')
    plt.show()
    plt.close()
else:
    print('Insufficient columns for enhanced pairplot (need area/bedrooms/bathrooms/price + location)')

# Comment: Enhanced pairplot visualizes pairwise relationships with location-based coloring to spot location-driven patterns.

# 15. Model Error Distribution (histogram of residuals: y_test - y_pred)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=60, kde=True, color='slateblue')
plt.title('Model Error Distribution (Residuals)', fontsize=13, fontweight='bold')
plt.xlabel('Residual (Actual - Predicted)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean={residuals.mean():.0f}')
plt.legend()
plt.tight_layout()
plt.savefig('advanced_plots/model_error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Comment: Histogram of residuals helps check bias and normality of prediction errors; KDE smooths the distribution.

# ========================================
# Summary of advanced plots
# ========================================
print('\n' + '='*50)
print('✓ Advanced visualizations completed and saved to advanced_plots/')
print('Files generated (examples):')
print('  - advanced_plots/count_categorical_features.png')
print('  - advanced_plots/train_vs_test_performance.png')
print('  - advanced_plots/3d_scatter_area_bedrooms_price.png')
print('  - advanced_plots/price_heatmap_map.html')
print('  - advanced_plots/interactive_area_price.html (and .png if supported)')
print('  - advanced_plots/enhanced_pairplot.png')
print('  - advanced_plots/model_error_distribution.png')
print('='*50)
