import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I was searching for interesting datasets and in Reddit I found https://www.airroi.com/, a page focused in Airbnb data analytics. It has some little sample datasets, and I decided to predict the *Number of booked/reserved days in trailing twelve months* for the available datasets of mexican cities with beaches:
    1. Acapulco.
    2. Bacalar.
    3. Cabo San Lucas.
    4. Cancun.
    5. Ensenada.
    6. Isla Mujeres.
    7. La Paz.
    8. Manzanillo.
    9. Mazatlan.
    10. Playa del Carmen.
    11. Puerto Escondido.
    12. Puerto Morelos.
    13. Puerto Vallarta.
    14. Rosarito.
    15. San Jose del Cabo.
    16. San Miguel de Cozumel.
    17. Sayulita.
    18. Tulum.

    One csv file for each city.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import holoviews as hv
    from holoviews import opts
    from holoviews.operation import gridmatrix
    import numpy as np
    import glob
    return glob, gridmatrix, hv, mo, np, opts, pd


@app.cell
def _(glob, mo, pd):
    # Get a list of all csv files in the './data' directory
    csv_files = glob.glob('./data/*.csv')

    # Check if any files were found before proceeding
    if csv_files:
        # Read all CSVs into a list of DataFrames and then concatenate them.
        # We use on_bad_lines='skip' to handle potential malformed rows.
        combined_df = pd.concat(
            [pd.read_csv(file, on_bad_lines='skip') for file in csv_files],
            ignore_index=True,
        )
    else:
        mo.md("No CSV files found in the `./data` directory.")
        combined_df = pd.DataFrame() # Create an empty dataframe
    combined_df
    return (combined_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4484 samples and 62 features. Lets do the EDA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data cleaning and "feature engineering" 🥸
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First of all I will see the data types of the dataset and its descriptive statistics:
    """)
    return


@app.cell
def _(combined_df):
    combined_df.info()
    return


@app.cell
def _(combined_df):
    combined_df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A lot of features are not listed using *combined_df.describe()* method. But *combined_df.info()* was helpful to identify missing data for some features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### First columns drop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I want to predict *ttm_reserved_days* using the descriptive features of the listing. So I decided to drop the following columns without EDA, good or bad decision? I will never know 🤣.


    The dropped features are:
    1. listing_name
    2. cover_photo_url
    3. host_id
    4. host_name
    5. cohost_ids
    6. cohost_names
    7. latitude
    8. longitude
    9. registration
    10. cancellation_policy
    11. currency

    I also include latitude and longituge because all are in Mexico, maybe in cities very far between them, but my analysis is at country level.

    And I will also dropped all *ttm* and l90 columns. Because I want to train my model with information when you are planning to open an Airbnb or at your start, not 12 or 3 months later.
    """)
    return


@app.cell
def _(combined_df):
    descriptive_features_df= combined_df.drop(columns=['listing_id', 'listing_name', 'cover_photo_url', 'host_id', 'host_name',
                                                       'cohost_ids','cohost_names', 'latitude','longitude',
                                                       'registration', 'currency', 'ttm_revenue', 'ttm_revenue_native', 
                                                       'ttm_avg_rate', 'ttm_avg_rate_native', 'ttm_occupancy', 'ttm_adjusted_occupancy',
                                                       'ttm_revpar', 'ttm_revpar_native', 'ttm_adjusted_revpar', 'ttm_adjusted_revpar_native', 
                                                       'ttm_blocked_days', 'ttm_available_days', 'ttm_total_days', 'l90d_revenue', 
                                                       'l90d_revenue_native', 'l90d_avg_rate', 'l90d_avg_rate_native', 'l90d_occupancy', 
                                                       'l90d_adjusted_occupancy', 'l90d_revpar', 'l90d_revpar_native', 'l90d_adjusted_revpar', 
                                                       'l90d_adjusted_revpar_native', 'l90d_reserved_days', 'l90d_blocked_days', 'l90d_available_days', 
                                                       'l90d_total_days'])
    return (descriptive_features_df,)


@app.cell
def _(descriptive_features_df):
    descriptive_features_df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _Listin_id_ because it is not a feature, it is an identifier. Probably I will use it later, so I will convert it in my index.
    """)
    return


@app.cell
def _():
    # descriptive_features_index = descriptive_features_df.set_index('listing_id')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why instant_book and professional_management are not booleans?

    These two are currently object type, so, lets check why and if is it possible to convert them into booleans.
    """)
    return


@app.cell
def _(descriptive_features_df):
    descriptive_features_df['instant_book'].value_counts(dropna=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I decied to drop 260 rows with null values and then convert the column as bool type.
    """)
    return


@app.cell
def _(descriptive_features_df):
    drop_instant_book_nulls = descriptive_features_df.dropna(subset=['instant_book'])
    drop_instant_book_nulls = drop_instant_book_nulls.astype({'instant_book': 'bool'})
    return (drop_instant_book_nulls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now _instant_book_ is a boolean feature!!!
    """)
    return


@app.cell
def _(drop_instant_book_nulls):
    drop_instant_book_nulls['professional_management'].value_counts(dropna=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _professional_management_ issue is more complex. It has 1184 null values, so I decided to drop this column to keep the analysis simple.
    """)
    return


@app.cell
def _(drop_instant_book_nulls):
    drop_professional_management = drop_instant_book_nulls.drop(columns='professional_management')
    return (drop_professional_management,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### _Rating_ columns with null values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For all the _rating_ columns with null values, I will fill them with 0. Assuming it is because they have very few number of reviews.
    """)
    return


@app.cell
def _(drop_professional_management):
    drop_professional_management['rating_overall'] = drop_professional_management['rating_overall'].fillna(0)
    drop_professional_management['rating_accuracy'] = drop_professional_management['rating_accuracy'].fillna(0)
    drop_professional_management['rating_checkin'] = drop_professional_management['rating_checkin'].fillna(0)
    drop_professional_management['rating_cleanliness'] = drop_professional_management['rating_cleanliness'].fillna(0)
    drop_professional_management['rating_communication'] = drop_professional_management['rating_communication'].fillna(0)
    drop_professional_management['rating_location'] = drop_professional_management['rating_location'].fillna(0)
    drop_professional_management['rating_value'] = drop_professional_management['rating_value'].fillna(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### _cleaning_fee_ and _extra_guest_fee_.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And something similar for _cleaning_fee_ and _extra_guest_fee_. Assumming null values as not fee.
    """)
    return


@app.cell
def _(drop_professional_management):
    drop_professional_management['cleaning_fee'] = drop_professional_management['cleaning_fee'].fillna(0)
    drop_professional_management['extra_guest_fee'] = drop_professional_management['extra_guest_fee'].fillna(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### _guests_, _bedrooms_, _beds_, and _min_nights_.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For _guests_, _bedrooms_, _beds_, and _min_nights_ I will fill null rows with them medians.
    """)
    return


@app.cell
def _(drop_professional_management):
    drop_professional_management['guests'] = drop_professional_management['guests'].fillna(drop_professional_management['guests'].median())
    drop_professional_management['bedrooms'] = drop_professional_management['bedrooms'].fillna(drop_professional_management['bedrooms'].median())
    drop_professional_management['beds'] = drop_professional_management['beds'].fillna(drop_professional_management['beds'].median())
    drop_professional_management['min_nights'] = drop_professional_management['min_nights'].fillna(drop_professional_management['min_nights'].median())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating amenities_count column.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And I want to create a new column for the number of amenities instead of the original string columns.
    """)
    return


@app.cell
def _(drop_professional_management):
    drop_professional_management['amenities_count'] = drop_professional_management['amenities'].str.split(',').str.len().astype(int)
    df = drop_professional_management.drop(columns=['amenities'])
    df.info()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(hv):
    hv.extension('bokeh')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TTM Reserved Days histogram
    """)
    return


@app.cell
def _(df, hv, np, opts):
    # 1. Original Distribution
    hist_original = hv.Histogram(
        np.histogram(df['ttm_reserved_days'], bins=50), 
        label='Original: TTM Reserved Days'
    ).opts(xlabel='Days Booked')

    # Combine side-by-side
    target_plot = (hist_original).opts(
        opts.Histogram(width=450, height=350, color='#1f77b4', tools=['hover'])
    )

    target_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From above histogram I can conclude that the _number of booked/reserved days in trailing twelve months_ is a good variable to be predicted because is distributed in a well way, having samples for a lot of different values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Correlation Matrix
    """)
    return


@app.cell
def _(df, hv):
    # Select numeric columns
    numeric_cols = ['photos_count', 'guests', 'bedrooms', 'beds', 'baths', 
                    'min_nights', 'cleaning_fee', 'extra_guest_fee', 'num_reviews', 
                    'rating_overall', 'rating_accuracy', 'rating_checkin', 'rating_cleanliness',
                    'rating_communication', 'rating_location', 'rating_value', 'amenities_count', 
                    'ttm_reserved_days']

    # Compute correlation and reshape for HoloViews
    corr_matrix = df[numeric_cols].corr()
    corr_tidy = corr_matrix.stack().reset_index()
    corr_tidy.columns = ['x', 'y', 'correlation']

    # Create HeatMap
    heatmap = hv.HeatMap(corr_tidy).opts(
        tools=['hover'],
        colorbar=True,
        width=700,
        height=600,
        toolbar='above',
        cmap='RdBu_r',          # Red-Blue colormap (Red = positive, Blue = negative)
        clim=(-1, 1),           # Lock color scale between -1 and 1
        xrotation=45,           # Rotate labels for readability
        title="Feature Correlation Matrix"
    )

    heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With above plot I decided to drop all _rating_ features _rating_overall_ because it is calculated from the other _ratings_ features.

    I also see that _baths_, _beds_, _bedrooms_ and _guest_ have a high correlation, but that is something obvious, because they are very related to the size of the house. In this case I decided to not drop any of them because in this case the dataset doesn´t have any other feature that summarize all of them.
    """)
    return


@app.cell
def _(df):
    df1 = df.drop(columns=['rating_value', 'rating_location', 'rating_communication', 'rating_cleanliness', 'rating_checkin', 'rating_accuracy' ])
    return (df1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Box plots of superhost starus and instant book.
    """)
    return


@app.cell
def _(df1, hv):
    df1['superhost'] = df1['superhost'].astype(str)
    df1['instant_book'] = df1['instant_book'].astype(str)
    # kdims = Categorical groupings (X-axis)
    # vdims = Continuous variable (Y-axis)
    box_plot = hv.BoxWhisker(
        df1, 
        kdims=['superhost', 'instant_book'], 
        vdims=['ttm_reserved_days']
    ).opts(
        width=600,
        height=400,
        box_fill_color='instant_book', # Color by the second category
        cmap='Set1',
        xlabel='Superhost Status / Instant Book',
        ylabel='Reserved Days',
        title='Impact of Trust & Friction on Bookings'
    )

    box_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both, _superhost_ and _instant_book_ appear not being influencing in _ttm_reserved_days_ but I am not sure if is good idea to drop them, I will leave them as part of the model.
    """)
    return


@app.cell(disabled=True)
def _(df1, gridmatrix, hv):
    #deactivating this cell to publish it to marimo service, but it render a beatiful plot.

    ds=hv.Dataset(df1)
    grouped_by_room_type= ds.groupby('cancellation_policy', container_type=hv.NdOverlay)
    grid = gridmatrix(grouped_by_room_type, diagonal_type=hv.Scatter)
    grid.options('Scatter',
                 width=120,   # Smaller width per square
                 height=120,  # Smaller height per square
                 fontsize={'labels': 8}, # Reduce font size if making plots small
                 tools=['hover', 'box_select'], 
                 bgcolor='#efe8e2', 
                 fill_alpha=0.2, 
                 size=4
                )
    return


@app.cell
def _(df1):
    df1['superhost'] = df1['superhost'].astype(bool)
    df1['instant_book'] = df1['instant_book'].astype(bool)
    df1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Training
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    return (
        ColumnTransformer,
        FunctionTransformer,
        GradientBoostingRegressor,
        OneHotEncoder,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        StandardScaler,
        mean_absolute_error,
        r2_score,
        train_test_split,
    )


@app.cell
def _():
    # ---------------------------------------------------------
    # 1. Setup Data (Matching Schema)
    # ---------------------------------------------------------

    target_col = 'ttm_reserved_days' # I am trying to predict this

    # Categorical features (Text)
    cat_features = ['listing_type', 'room_type', 'cancellation_policy']

    # Boolean features (True/False)
    bool_features = ['superhost', 'instant_book']

    # Numerical features (Everything else)
    num_features = [
        'photos_count', 'guests', 'bedrooms', 'beds', 'baths', 
        'min_nights', 'cleaning_fee', 'extra_guest_fee', 'num_reviews', 
        'rating_overall', 'amenities_count'
    ]
    return bool_features, cat_features, num_features, target_col


@app.cell
def _(
    ColumnTransformer,
    FunctionTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    bool_features,
    cat_features,
    num_features,
):
    # ---------------------------------------------------------
    # 2. Preprocessing Pipelines
    # ---------------------------------------------------------

    # Pipeline for Numerical Data:
    # 1. Fill missing values with the Median (handles the missing 'num_reviews')
    # 2. Scale data (StandardScaler) helps models converge faster
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for Categorical Data:
    # 1. OneHotEncode converts "Apartment" -> [0, 1, 0]
    # handle_unknown='ignore' prevents crashes if new categories appear in the future
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Pipeline for Boolean Data: Cast to Float -> Fill missing
    # This converts True->1.0 and False->0.0
    boolean_transformer = Pipeline(steps=[
        ('cast_to_float', FunctionTransformer(lambda x: x.astype(float), validate=False)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # Combine them into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
            ('bool', boolean_transformer, bool_features) # <--- Use the new transformer here
        ])
    return (preprocessor,)


@app.cell
def _(
    GradientBoostingRegressor,
    Pipeline,
    RandomForestRegressor,
    Ridge,
    df1,
    mean_absolute_error,
    preprocessor,
    r2_score,
    target_col,
    train_test_split,
):
    # ---------------------------------------------------------
    # 3. Define the Model
    # ---------------------------------------------------------

    models = {
        "Ridge Regression (Linear) - Alpha = 1.0": Ridge(alpha=1.0),
        "Ridge Regression (Linear) - Alpha = 5.0": Ridge(alpha=5.0),
        "Ridge Regression (Linear) - Alpha = 10.0": Ridge(alpha=10.0),
        "Random Forest - n_estimators = 100": RandomForestRegressor(n_estimators=100, random_state=42),
        "Random Forest - n_estimators = 500": RandomForestRegressor(n_estimators=500, random_state=42),
        "Random Forest - n_estimators = 1000": RandomForestRegressor(n_estimators=1000, random_state=42),
        "Gradient Boosting - n_estimators = 100, learning_rate = 0.1": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Gradient Boosting - n_estimators = 500, learning_rate = 0.1": GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, random_state=42),
        "Gradient Boosting - n_estimators = 1000, learning_rate = 0.1": GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, random_state=42),
        "Gradient Boosting - n_estimators = 100, learning_rate = 0.5": GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, random_state=42),
        "Gradient Boosting - n_estimators = 500, learning_rate = 0.5": GradientBoostingRegressor(n_estimators=500, learning_rate=0.5, random_state=42),
        "Gradient Boosting - n_estimators = 1000, learning_rate = 0.5": GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5, random_state=42),
        "Gradient Boosting - n_estimators = 100, learning_rate = 1.0": GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, random_state=42),
        "Gradient Boosting - n_estimators = 500, learning_rate = 1.0": GradientBoostingRegressor(n_estimators=500, learning_rate=1.0, random_state=42),
        "Gradient Boosting - n_estimators = 1000, learning_rate = 1.0": GradientBoostingRegressor(n_estimators=1000, learning_rate=1.0, random_state=42),
    }

    print(f"{'Model Name':<30} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 55)

    X = df1.drop(columns=[target_col])
    y = df1[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model_instance in models.items():
        # Create a new pipeline with the specific model
        # (Assuming 'preprocessor' is defined from the previous steps)
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_instance)
        ])

        # Train
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{name:<30} | {mae:<10.2f} | {r2:<10.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The best model is Gradient Boosting - n_estimators = 100, learning_rate = 0.1 because it has the lowest MAE (45.46) and the highest R2 Score (0.3133)**
    """)
    return


if __name__ == "__main__":
    app.run()
