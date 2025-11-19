import pandas as pd
glob_imported = __import__('glob')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 1. Load and combine CSVs
data_path = './data/*.csv'
csv_files = glob_imported.glob(data_path)
if csv_files:
    combined_df = pd.concat([
        pd.read_csv(file, on_bad_lines='skip') for file in csv_files
    ], ignore_index=True)
else:
    raise FileNotFoundError("No CSV files found in the './data' directory.")

# 2. Drop columns
columns_to_drop = [
    'listing_id', 'listing_name', 'cover_photo_url', 'host_id', 'host_name',
    'cohost_ids','cohost_names', 'latitude','longitude',
    'registration', 'currency', 'ttm_revenue', 'ttm_revenue_native',
    'ttm_avg_rate', 'ttm_avg_rate_native', 'ttm_occupancy', 'ttm_adjusted_occupancy',
    'ttm_revpar', 'ttm_revpar_native', 'ttm_adjusted_revpar', 'ttm_adjusted_revpar_native',
    'ttm_blocked_days', 'ttm_available_days', 'ttm_total_days', 'l90d_revenue',
    'l90d_revenue_native', 'l90d_avg_rate', 'l90d_avg_rate_native', 'l90d_occupancy',
    'l90d_adjusted_occupancy', 'l90d_revpar', 'l90d_revpar_native', 'l90d_adjusted_revpar',
    'l90d_adjusted_revpar_native', 'l90d_reserved_days', 'l90d_blocked_days', 'l90d_available_days',
    'l90d_total_days'
]
descriptive_features_df = combined_df.drop(columns=columns_to_drop)

# 3. Convert instant_book to bool, drop nulls
if 'instant_book' in descriptive_features_df.columns:
    descriptive_features_df = descriptive_features_df.dropna(subset=['instant_book'])
    descriptive_features_df['instant_book'] = descriptive_features_df['instant_book'].astype(bool)

# 4. Drop professional_management
if 'professional_management' in descriptive_features_df.columns:
    descriptive_features_df = descriptive_features_df.drop(columns='professional_management')

# 5. Fill rating columns with 0
for col in ['rating_overall', 'rating_accuracy', 'rating_checkin', 'rating_cleanliness', 'rating_communication', 'rating_location', 'rating_value']:
    if col in descriptive_features_df.columns:
        descriptive_features_df[col] = descriptive_features_df[col].fillna(0)

# 6. Fill cleaning_fee and extra_guest_fee with 0
for col in ['cleaning_fee', 'extra_guest_fee']:
    if col in descriptive_features_df.columns:
        descriptive_features_df[col] = descriptive_features_df[col].fillna(0)

# 7. Fill guests, bedrooms, beds, min_nights with median
for col in ['guests', 'bedrooms', 'beds', 'min_nights']:
    if col in descriptive_features_df.columns:
        descriptive_features_df[col] = descriptive_features_df[col].fillna(descriptive_features_df[col].median())

# 8. Create amenities_count and drop amenities
if 'amenities' in descriptive_features_df.columns:
    descriptive_features_df['amenities_count'] = descriptive_features_df['amenities'].str.split(',').str.len().astype(int)
    descriptive_features_df = descriptive_features_df.drop(columns=['amenities'])

df = descriptive_features_df.copy()

# 9. Drop additional rating columns after correlation analysis
drop_ratings = ['rating_value', 'rating_location', 'rating_communication', 'rating_cleanliness', 'rating_checkin', 'rating_accuracy']
df1 = df.drop(columns=drop_ratings)

# 10. Convert superhost and instant_book to bool
for col in ['superhost', 'instant_book']:
    if col in df1.columns:
        df1[col] = df1[col].astype(bool)

# 11. Drop nulls values of num_reviews
if 'instant_book' in descriptive_features_df.columns:
    df1 = df1.dropna(subset=['num_reviews'])

# 11. Setup features
cat_features = ['listing_type', 'room_type', 'cancellation_policy']
bool_features = ['superhost', 'instant_book']
num_features = [
    'photos_count', 'guests', 'bedrooms', 'beds', 'baths',
    'min_nights', 'cleaning_fee', 'extra_guest_fee', 'num_reviews',
    'rating_overall', 'amenities_count'
]
target_col = 'ttm_reserved_days'


# 13. Preprocessing pipelines
numeric_transformer = FunctionTransformer(validate=False)

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


def cast_to_float(x): # Custom function to cast boolean to float, if I use a lambda inside FunctionTransformer it fails with pickling.
    return x.astype(float)

boolean_transformer = Pipeline([
    ('cast_to_float', FunctionTransformer(cast_to_float, validate=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features),
    ('bool', boolean_transformer, bool_features)
])

# 14. Model training and evaluation
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# 15. Partition data
X = df1.drop(columns=[target_col])
y = df1[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 16. Train best model
best_model.fit(X_train, y_train)

# 17. Save the best model as a pickle file
joblib.dump(best_model, './gradient_boosting_model.bin')