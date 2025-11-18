import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


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
    import numpy as np
    import glob
    return glob, hv, mo, np, opts, pd


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
    descriptive_features_df= combined_df.drop(columns=['listing_name', 'cover_photo_url', 'host_id', 'host_name',
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
def _(descriptive_features_df):
    descriptive_features_index = descriptive_features_df.set_index('listing_id')
    return (descriptive_features_index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why instant_book and professional_management are not booleans?

    These two are currently object type, so, lets check why and if is it possible to convert them into booleans.
    """)
    return


@app.cell
def _(descriptive_features_index):
    descriptive_features_index['instant_book'].value_counts(dropna=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I decied to drop 260 rows with null values and then convert the column as bool type.
    """)
    return


@app.cell
def _(descriptive_features_index):
    drop_instant_book_nulls = descriptive_features_index.dropna(subset=['instant_book'])
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
    drop_professional_management['amenities count'] = drop_professional_management['amenities'].str.split(',').str.len().astype(int)
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


@app.cell
def _(df, hv):
    # Select numeric columns
    numeric_cols = ['photos_count', 'guests', 'bedrooms', 'beds', 'baths', 
                    'min_nights', 'cleaning_fee', 'num_reviews', 'rating_overall', 
                    'amenities count', 'ttm_reserved_days']

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


@app.cell
def _(df, hv):
    # kdims = Categorical groupings (X-axis)
    # vdims = Continuous variable (Y-axis)
    box_plot = hv.BoxWhisker(
        df, 
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


if __name__ == "__main__":
    app.run()
