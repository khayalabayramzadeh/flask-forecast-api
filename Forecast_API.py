from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pymysql

app = Flask(__name__)

# SQL bağlantısı
host = '38.242.245.151'
port = 3306
database = 'ANALYTICS'
username = 'ANALYTICS_USER'
password = 'E223YI3487356231eST@-'

# Saxlanmış nəticələri saxlamaq üçün qlobal dəyişən
stored_forecast = None

# SQL-dən məlumatları almaq funksiyası
def get_data():
    cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)
    query = """
    SELECT 
        REGION,
        IL,
        RUB2 AS RUB,
        MEHSUL_KODU,
        SUM(MIQDAR) AS MIQDAR
    FROM ANALYTICS.SUPER_SALE2_VR
    GROUP BY
        REGION,
        IL,
        RUB2,
        MEHSUL_KODU
    ORDER BY
        REGION,
        IL,
        RUB2,
        MEHSUL_KODU
    """      
    data = pd.read_sql(query, cnxn)
    cnxn.close()
    return data

# Proqnoz üçün əsas funksiyanız
def forecast_sales_data(data, target_year, target_rub):
    def forecast_sales(product_data):
        product_data['IL'] = product_data['IL'].astype(int)
        product_data['RUB'] = product_data['RUB'].astype(int)
        
        # X (rüblər) və Y (satışlar) məlumatlarını hazırlamaq
        quarters = product_data[['IL', 'RUB']].apply(lambda row: (row['IL'] - 2022) * 3 + row['RUB'], axis=1).values
        sales = product_data['MIQDAR'].values

        if len(quarters) < 7:
            x = quarters.reshape(-1, 1)
            y = sales
        else:
            x = quarters[-7:].reshape(-1, 1)
            y = sales[-7:]

        # Modelin təlimi
        model = LinearRegression()
        model.fit(x, y)
        
        return model

    def gradual_forecast(data, target_year, target_rub):
        data_grouped = data.groupby(['REGION', 'MEHSUL_KODU', 'IL', 'RUB']).agg({'MIQDAR': 'sum'}).reset_index()

        forecast_results = []
        grouped_data = data_grouped.groupby(['REGION', 'MEHSUL_KODU'])
        
        for (region, product), group in grouped_data:
            model = forecast_sales(group)

            future_data = group[(group['IL'] < target_year) | ((group['IL'] == target_year) & (group['RUB'] < target_rub))]
            future_quarters = future_data[['IL', 'RUB']].apply(lambda row: (row['IL'] - 2022) * 3 + row['RUB'], axis=1).values
            future_sales = future_data['MIQDAR'].values
            
            if len(future_sales) > 0:
                x_real = future_quarters.reshape(-1, 1)
                y_real = future_sales
                model.fit(x_real, y_real)

            for year in range(2023, target_year + 1):
                for rub in range(1, 4):
                    if (year == target_year and rub > target_rub):
                        break

                    if (year, rub) not in zip(group['IL'], group['RUB']):
                        future_quarter = np.array([[(year - 2022) * 3 + rub]]).reshape(-1, 1)
                        forecast = model.predict(future_quarter)[0]

                        new_row = pd.DataFrame([{'IL': year, 'RUB': rub, 'MIQDAR': forecast}])
                        future_data = pd.concat([future_data, new_row], ignore_index=True)

                        future_quarters = future_data[['IL', 'RUB']].apply(lambda row: (row['IL'] - 2022) * 3 + row['RUB'], axis=1).values
                        future_sales = future_data['MIQDAR'].values
                        x_new = future_quarters.reshape(-1, 1)
                        y_new = future_sales
                        model.fit(x_new, y_new)

            future_quarter = np.array([[(target_year - 2022) * 3 + target_rub]]).reshape(-1, 1)
            forecast = model.predict(future_quarter)[0]

            forecast_results.append({
                'REGION': region,
                'MEHSUL_KODU': product,
                'Forecast_{}_Q{}'.format(target_year, target_rub): forecast
            })

        forecast_df = pd.DataFrame(forecast_results)
        return forecast_df

    forecast_df = gradual_forecast(data, target_year, target_rub)
    return forecast_df

# 1-ci API: İl və rübə görə proqnoz nəticələrini almaq və saxlamaq
@app.route('/get_forecast', methods=['POST'])
def get_forecast():
    global stored_forecast
    content = request.json
    target_year = content.get('target_year', 2024)
    target_rub = content.get('target_rub', 2)

    # SQL-dən məlumatları al
    data = get_data()

    # İl və rübə əsasən proqnozu al
    forecast_df = forecast_sales_data(data, target_year, target_rub)

    # Nəticəni qlobal dəyişəndə saxla
    stored_forecast = forecast_df.copy()

    # Proqnoz məlumatlarını JSON formatında qaytar
    return jsonify(forecast_df.to_dict(orient='records'))

# 2-ci API: Saxlanmış proqnoz nəticələrini region və məhsul koduna görə filtr et
@app.route('/filter_forecast', methods=['POST'])
def filter_forecast():
    global stored_forecast
    if stored_forecast is None:
        return jsonify({"error": "Forecast data is not available. Please call the /get_forecast API first."})

    content = request.json
    selected_region = content.get('selected_region')
    selected_product = content.get('selected_product')

    # Saxlanmış proqnozdan region və məhsul kodu üzrə filtr et
    filtered_result = stored_forecast[
        (stored_forecast['REGION'] == selected_region) &
        (stored_forecast['MEHSUL_KODU'] == selected_product)
    ]

    # Filtrlənmiş nəticəni JSON formatında qaytar
    return jsonify(filtered_result.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
