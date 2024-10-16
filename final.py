from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pymysql
import json
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# SQL connection details
host = '38.242.245.151'
port = 3306
database = 'ANALYTICS'
username = 'ANALYTICS_USER'
password = 'E223YI3487356231eST@-'    

# Function to retrieve data from the SQL database
def get_data():
    retries = 3
    for attempt in range(retries):
        try:
            cnxn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database,
                connect_timeout=120,  
                read_timeout=120     
            )
            query = """
            SELECT * FROM ANALYTICS.FORECASTS_DETAILS
            WHERE IS_ACTIVE=1;
            """
            data = pd.read_sql(query, cnxn)
            cnxn.close()
            return data
        except pymysql.MySQLError as e:
            if attempt < retries - 1:
                logging.warning(f"Query failed, retrying... ({attempt + 1}/{retries})")
                time.sleep(5)  
            else:
                logging.error(f"Database connection error: {str(e)}")
                return {"error": f"Database connection error: {str(e)}"}



# Function to insert forecast results in the database
def insert_forecast_requests(year, rub):
    try:
        cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)
        cursor = cnxn.cursor()

        logging.debug(f"Inserting new record for Year {year} and Rub {rub}.")
        insert_query = """
        INSERT INTO ANALYTICS.FORECAST_REQUEST (YEAR, RUB)
        VALUES (%s, %s)
        """
        cursor.execute(insert_query, (year, rub))
        cnxn.commit()

        cursor.close()
        cnxn.close()

    except Exception as e:
        logging.error(f"Error during database operation: {str(e)}")


# Function to insert forecast results into the database without duplication
def insert_forecast_results(results):
    try:
        cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)
        cursor = cnxn.cursor()

        logging.debug(f"Inserting forecast results in bulk, checking for duplicates.")
        insert_query = """
        INSERT INTO ANALYTICS.FORECASTS_DETAILS (REGION, MEHSUL_KODU, IL, RUB, MIQDAR)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE MIQDAR = VALUES(MIQDAR)
        """

        # Convert lists to tuples for uniqueness check
        unique_results = list(set(tuple(result) for result in results))
        
        cursor.executemany(insert_query, unique_results)
        cnxn.commit()

        cursor.close()
        cnxn.close()

    except Exception as e:
        logging.error(f"Error during database operation: {str(e)}")



# Function to check if existing data is present in FORECASTS_DETAILS
def check_existing_data(year, rub):
    try:
        cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)

        query = """
        SELECT FORECAST_RESULTS_ID FROM ANALYTICS.FORECAST_RESULTS
        WHERE YEAR = %s AND RUB = %s AND IS_ACTIVE=1
        """
        data = pd.read_sql(query, cnxn, params=[year, rub])
        
        cnxn.close()

        if not data.empty:
            logging.debug(f"Existing data found for IL = {year} and RUB = {rub}. Returning FORECAST_RESULTS_ID.")
            forecast_results_id = data['FORECAST_RESULTS_ID'].iloc[0]
            details = get_forecasts_details(forecast_results_id)
            return details
        
        return None

    except Exception as e:
        logging.error(f"Error during database operation: {str(e)}")
        return None 

def get_forecasts_details(forecast_results_id):
    try:
        cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)

        query = """
        SELECT * FROM ANALYTICS.FORECASTS_DETAILS
        WHERE FORECAST_RESULTS_ID = %s
        """
        data = pd.read_sql(query, cnxn, params=[forecast_results_id])
        
        cnxn.close()

        return data

    except Exception as e:
        logging.error(f"Error during database operation: {str(e)}")
        return None

# Function to insert missing and target quarters
def insert_missing_and_target_quarters(missing_quarters, target_year, target_rub):
    try:
        cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)
        cursor = cnxn.cursor()

        insert_query = """
        INSERT INTO ANALYTICS.FORECAST_RESULTS (YEAR, RUB)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
        YEAR = VALUES(YEAR), RUB = VALUES(RUB)
        """
        
        # Target il və rübün artıq listdə olub-olmadığını yoxlayırıq
        all_quarters = missing_quarters + [(target_year, target_rub)]
        all_quarters = list(set(all_quarters))  # Dublikatları silirik
        
        cursor.executemany(insert_query, all_quarters)
        cnxn.commit()

        cursor.close()
        cnxn.close()

    except Exception as e:
        logging.error(f"Error inserting missing quarters: {str(e)}")

# Function to update FORECAST_RESULTS_ID in FORECASTS_DETAILS after forecast completion
def update_forecast_results_id():
    try:
        cnxn = pymysql.connect(host=host, port=port, user=username, password=password, database=database)
        cursor = cnxn.cursor()

        logging.debug("Updating FORECAST_RESULTS_ID in FORECASTS_DETAILS after forecast completion.")

        update_query = """
        UPDATE ANALYTICS.FORECASTS_DETAILS 
        SET FORECAST_RESULTS_ID = (
            SELECT R.FORECAST_RESULTS_ID 
            FROM ANALYTICS.FORECAST_RESULTS R
            WHERE ANALYTICS.FORECASTS_DETAILS.IL = R.YEAR 
              AND ANALYTICS.FORECASTS_DETAILS.RUB = R.RUB 
              AND R.IS_ACTIVE = 1
        )
        WHERE FORECAST_RESULTS_ID IS NULL
        """
        
        cursor.execute(update_query)
        cnxn.commit()

        cursor.close()
        cnxn.close()

    except Exception as e:
        logging.error(f"Error updating FORECAST_RESULTS_ID: {str(e)}")



# Forecasting function
def forecast_sales_data(data, target_year, target_rub):
    logging.debug("Starting sales forecasting process.")

    def determine_required_quarters(target_year, target_rub):
        required_quarters = []
        for year in range(target_year - 2, target_year + 1):
            for rub in range(1, 4):
                if (year == target_year and rub > target_rub):
                    break
                required_quarters.append((year, rub))
        return required_quarters

    def check_active_quarters_globally(required_quarters):
        active_quarters = []
        for year, rub in required_quarters:
            is_active = query_forecast_results_globally(year, rub)
            if is_active:
                active_quarters.append((year, rub))
        return active_quarters

    def forecast_missing_quarters(data, missing_quarters, region, product):
        group_data = data[(data['REGION'] == region) & (data['MEHSUL_KODU'] == product)]
        for year, rub in missing_quarters:
            forecast_data = group_data[(group_data['IL'] == year) & (group_data['RUB'] == rub)]
            if forecast_data.empty:
                model = forecast_sales(group_data)
                future_quarter = np.array([[(year - 2022) * 3 + rub]]).reshape(-1, 1)
                forecast = model.predict(future_quarter)[0]
                new_row = pd.DataFrame([{'IL': year, 'RUB': rub, 'REGION': region, 'MEHSUL_KODU': product, 'MIQDAR': forecast}])
                data = pd.concat([data, new_row], ignore_index=True)
        return data

    required_quarters = determine_required_quarters(target_year, target_rub)
    active_quarters = check_active_quarters_globally(required_quarters)

    missing_quarters = list(set(required_quarters) - set(active_quarters))

    data_grouped = data.groupby(['REGION', 'MEHSUL_KODU'])
    forecast_results = []

    for (region, product), group in data_grouped:
        logging.debug(f"Processing REGION: {region}, MEHSUL_KODU: {product}.")

        if missing_quarters:
            data = forecast_missing_quarters(data, missing_quarters, region, product)

        model = forecast_sales(group)
        future_quarter = np.array([[(target_year - 2022) * 3 + target_rub]]).reshape(-1, 1)
        forecast = model.predict(future_quarter)[0]

        forecast_results.append((region, product, target_year, target_rub, forecast))

        for year, rub in missing_quarters:
            future_quarter_missing = np.array([[(year - 2022) * 3 + rub]]).reshape(-1, 1)
            forecast_missing = model.predict(future_quarter_missing)[0]
            forecast_results.append((region, product, year, rub, forecast_missing))

    insert_missing_and_target_quarters(missing_quarters, target_year, target_rub)
    
    forecast_df = pd.DataFrame(forecast_results, columns=['REGION', 'MEHSUL_KODU', 'YEAR', 'RUB', 'MIQDAR'])
    logging.debug("Sales forecasting process complete.")
    return forecast_df

# Function to query active forecast results globally
def query_forecast_results_globally(year, rub):
    try:
        connection = pymysql.connect(host=host, port=port, user=username, password=password, database=database)
        with connection.cursor() as cursor:
            sql_query = """
                SELECT IS_ACTIVE 
                FROM ANALYTICS.FORECAST_RESULTS 
                WHERE YEAR = %s AND RUB = %s
            """
            cursor.execute(sql_query, (year, rub))
            result = cursor.fetchone()
            if result and result[0] == 1:
                return True
            else:
                return False
    except Exception as e:
        logging.error(f"An error occurred while querying the database: {e}")
        return False
    finally:
        if connection:
            connection.close()

# Function to forecast sales for product data
def forecast_sales(product_data):
    product_data['IL'] = product_data['IL'].astype(int)
    product_data['RUB'] = product_data['RUB'].astype(int)

    quarters = product_data[['IL', 'RUB']].apply(lambda row: (row['IL'] - 2022) * 3 + row['RUB'], axis=1).values
    sales = product_data['MIQDAR'].values

    if len(quarters) < 7:
        model = LinearRegression()
    else:
        model = LinearRegression()

    model.fit(quarters.reshape(-1, 1), sales)
    return model

# Define API endpoint
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        target_year = int(data.get('target_year'))
        target_rub = int(data.get('target_rub'))



        if target_year is None or target_rub is None:
          logging.error("Both target_year and target_rub must be provided.")
          return jsonify({"error": "Both target_year and target_rub must be provided."}), 400

        if target_year < 2022:
          logging.error("target_year must be 2022 or greater.")
          return jsonify({"error": "target_year must be 2022 or greater."}), 400

        if target_rub < 1 or target_rub > 3:
           logging.error("target_rub must be between 1 and 3.")
           return jsonify({"error": "target_rub must be between 1 and 3."}), 400



        insert_forecast_requests(target_year, target_rub)

        existing_data = check_existing_data(target_year, target_rub)
        if existing_data is not None:
            logging.debug("Returning existing forecast results.")
            return jsonify(existing_data.to_dict(orient='records'))

        full_data = get_data()
        if 'error' in full_data:
            return jsonify(full_data), 500

        forecasted_data = forecast_sales_data(full_data, target_year, target_rub)

        forecast_results = forecasted_data[['REGION', 'MEHSUL_KODU', 'YEAR', 'RUB', 'MIQDAR']].values.tolist()    
        insert_forecast_results(forecast_results)
        update_forecast_results_id()
        
        return jsonify(forecasted_data.to_dict(orient='records'))

    except Exception as e:
        logging.error(f"Error occurred in /forecast endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)