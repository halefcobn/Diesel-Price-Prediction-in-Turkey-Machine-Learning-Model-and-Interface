import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, mean_squared_error, r2_score
import numpy as np
from PIL import Image

# Function to preprocess the user-uploaded data
def preprocess_data(df):
    df.rename(columns={
        'Value-Added Tax (KDV)(%)': 'KDV',
        'Excise Duty (OTV-TL/l)': 'OTV',
        'USD / TRY rate(1$=TL)': 'USD',
        'Consumer Price Index (TUFE-Monthly Diff)': 'TUFE',
        'Turkey Diesel Usage (Tons)': 'USAGE',
        'Crude Oil Price (Brent-$)': 'CRUDE OIL',
        'Pump Selling Price(TL PER LİTER)': 'PRICE',
    }, inplace=True)

    df['USAGE'] = df['USAGE'].str.replace(',', '').astype(float)
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to datetime.date without time component

    return df[['Date', 'KDV', 'OTV', 'USD', 'TUFE', 'USAGE', 'CRUDE OIL']], df['PRICE']

# Streamlit app
#st.image('./.png', width=200)

st.title('Diesel Price Predictor')

# File uploader for user's dataset
st.write("Upload your dataset (Excel or CSV) / Excel veya CSV verisini yükleyin")
uploaded_file = st.file_uploader("Choose a file / Bir dosya seçin", type=["xlsx", "csv"])

if uploaded_file is not None:
    user_data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)

    # Preprocess the user-uploaded data
    X_user, y_user = preprocess_data(user_data)

    # Calendar for date selection
    selected_date = st.sidebar.date_input(
        "Select a date / Bir tarih seçiniz",
        min_value=X_user['Date'].min(),
        max_value=X_user['Date'].max(),
        value=X_user['Date'].max()
    )

    # Display input values based on the selected date
# Display input values based on the selected date
    selected_data = X_user[X_user['Date'] == selected_date]
    selected_data = selected_data[['Date','KDV', 'OTV', 'USD', 'TUFE', 'USAGE', 'CRUDE OIL']].dropna()  # Drop rows with missing values

    if not selected_data.empty:
        table_str = selected_data.to_markdown(index=False)  # Convert DataFrame to markdown string without the index column
        st.markdown(table_str)
    else:
        st.warning('No data available for the selected date.')

    # Exclude 'Date' column from training features
    X_train = X_user.drop(columns=['Date'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_user, test_size=0.3, random_state=42)

    # Train the RandomForestRegressor model on the training set
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    # Sidebar for input variables
    st.sidebar.write('### Input Variables / Girilmesi Gereken Veriler')
    kdv = st.sidebar.number_input('Value-Added Tax / KDV (%)', min_value=0.0)
    otv = st.sidebar.number_input('Excise Duty / OTV', min_value=0.0)
    usd = st.sidebar.number_input('USD / TRY rate(1$=TL?) / Dolar Kuru', min_value=0.0)
    tufe = st.sidebar.number_input('Consumer Price Index / TUFE', min_value=0.0)
    usage = st.sidebar.number_input('Turkey Diesel Usage / Türkiye Motorin Kullanımı (Tons) ', min_value=0.0)
    crude_oil = st.sidebar.number_input('Crude Oil Price (Brent-$) / Brent Ham Petrol Fiyatı', min_value=0.0)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'KDV': [kdv],
        'OTV': [otv],
        'USD': [usd],
        'TUFE': [tufe],
        'USAGE': [usage],
        'CRUDE OIL': [crude_oil]
    })

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('Predict Diesel Price / Motorin Fiyat Tahmini için Butona Basınız'):
        # Predict the price
        if kdv == otv == usd == tufe == usage == crude_oil == 0:
            st.warning('Please provide accurate input values / Lütfen veri giriniz')
        else:
            predicted_price = rf_model.predict(input_data)
            st.subheader(f'Predicted Price / Tahmin Edilen Fiyat: {predicted_price[0]:.2f}')

            # Calculate metrics on the testing set
            predicted_price_test = rf_model.predict(X_test)
            mae_test = mean_absolute_error(y_test, predicted_price_test)
            mape_test = mean_absolute_percentage_error(y_test, predicted_price_test)
            evs_test = explained_variance_score(y_test, predicted_price_test)
            mse_test = mean_squared_error(y_test, predicted_price_test)
            rmse_test = np.sqrt(mse_test)
     
            r2_test = r2_score(y_test, predicted_price_test)

            st.write('### Metrics on the Testing Set: / Yüklenen Verinin Doğruluk Performansı ')
            st.write(f'Mean Absolute Error (MAE): {mae_test:.3f}')
            st.write(f'Mean Absolute Percentage Error (MAPE): {mape_test:.3f}')
            st.write(f'Explained Variance Score: {evs_test:.3f}')
            st.write(f'Mean Squared Error (MSE): {mse_test:.3f}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse_test:.3f}')
            st.write(f'R-squared (R²): {r2_test:.3f}')
