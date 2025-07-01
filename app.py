from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from mftool import Mftool
from decimal import Decimal 
from statsmodels.tsa.arima.model import ARIMA
import os
from config import db_connection, get_dict_cursor
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
mf = Mftool()
INDIAN_HOLIDAYS = {
    '01-01', '26-01', '15-08', '02-10', '25-12', '29-03', '14-04', '01-05'
}

if not os.path.exists("static/images"):
    os.makedirs("static/images")

def recommend_action(forecast):
    last_nav = forecast.iloc[-2]
    next_nav = forecast.iloc[-1]

    if next_nav > last_nav:
        return "BUY ✅ - Expected Growth", "The NAV is predicted to increase, indicating potential growth. It might be a good opportunity to invest."
    elif next_nav < last_nav:
        return "SELL ❌ - Expected Drop", "The NAV is predicted to decrease, suggesting a decline. You might consider selling or holding off investments."
    else:
        return "HOLD ⚖️ - No Significant Change", "There are no significant fluctuations in NAV. Holding the investment may be a stable choice."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_nav', methods=['POST'])
def get_nav():
    scheme_code = request.form['scheme_code']
    nav_data = mf.get_scheme_historical_nav(scheme_code)

    try:
        fund_details = mf.calculate_returns(
            code=scheme_code, balanced_units=1, monthly_sip=1, investment_in_months=1
        )
    except ZeroDivisionError:
        return render_template('index.html', error="Calculation Error: Initial investment cannot be zero.")

    if nav_data and 'data' in nav_data and fund_details:
        df = pd.DataFrame(nav_data['data'])
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = df['nav'].astype(float)
        df = df.sort_values('date')

        conn = db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO funds (scheme_code, scheme_name, last_nav, last_updated, 
                               absolute_return, irr_annualised_return, final_investment_value) 
            VALUES (%s, %s, %s, NOW(), %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                last_nav = VALUES(last_nav), 
                last_updated = NOW(),
                scheme_name = VALUES(scheme_name),
                absolute_return = VALUES(absolute_return),
                irr_annualised_return = VALUES(irr_annualised_return),
                final_investment_value = VALUES(final_investment_value)
            """,
            (
                scheme_code, 
                fund_details.get('scheme_name', 'Unknown'),
                df['nav'].iloc[-1],
                fund_details.get('absolute_return', 0),
                fund_details.get('IRR_annualised_return', 0),
                fund_details.get('final_investment_value', 0)
            )
        )
        conn.commit()
        cursor.close()
        conn.close()

        model = ARIMA(df['nav'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)

        future_dates = pd.date_range(start=df['date'].iloc[-1], periods=31, freq='D')[1:]

        action, description = recommend_action(forecast)

        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['nav'], label='Historical NAV', color='blue')
        plt.plot(future_dates, forecast, label='Predicted NAV', linestyle='dashed', color='red')
        plt.xlabel('Date')
        plt.ylabel('NAV Price')
        plt.title(f'Predicted NAV for Scheme Code {scheme_code}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        img_path = f'static/images/nav_{scheme_code}.png'
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        return render_template(
            'index.html',
            img_path=img_path,
            scheme_code=scheme_code,
            action=action,
            description=description,
            fund_details=fund_details
        )

    return render_template('index.html', error="Invalid Scheme Code or No Data Found")

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    try:
        data = request.json
        scheme_code = data.get('scheme_code')
        amount = Decimal(str(data.get('amount', 0)))
        input_date = data.get('date')

        if amount <= 0:
            return jsonify({"error": "Amount must be greater than zero."}), 400

        # Convert input date
        try:
            transaction_date = datetime.strptime(input_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        print(f"Original Transaction Date: {transaction_date.strftime('%Y-%m-%d')}")

        nav_data = mf.get_scheme_historical_nav(scheme_code)

        if not nav_data or 'data' not in nav_data:
            return jsonify({"error": "Invalid Scheme Code or No Data Found"}), 400

        def get_nav_for_date(date):
            return next(
                (Decimal(str(item['nav'])) for item in nav_data['data']
                 if item['date'] == date.strftime('%d-%m-%Y')),
                None
            )

        # Step 1: Try fetching NAV for the original date
        latest_nav = get_nav_for_date(transaction_date)

        # Step 2: If unavailable, check for weekend or holiday
        if (transaction_date.weekday() in [5, 6] or  # Weekend
            transaction_date.strftime('%d-%m') in INDIAN_HOLIDAYS or
            not latest_nav):
            
            print(f"{transaction_date.strftime('%Y-%m-%d')} is a holiday/weekend or NAV unavailable.")

            # Step 3: Try the same date and month from the previous year
            previous_year_date = transaction_date.replace(year=transaction_date.year - 1)
            latest_nav = get_nav_for_date(previous_year_date)

            if not latest_nav:
                # Step 4: If still not available, roll back day-by-day
                while not latest_nav:
                    print(f"No NAV for {transaction_date.strftime('%Y-%m-%d')}. Rolling back...")
                    transaction_date -= timedelta(days=1)
                    latest_nav = get_nav_for_date(transaction_date)

        if not latest_nav:
            return jsonify({"error": "No valid NAV available after adjustments."}), 400

        units = round(amount / latest_nav, 4)
        print(f"Final Transaction Date: {transaction_date.strftime('%Y-%m-%d')}")
        print(f"NAV: {latest_nav}, Units: {units}")

        # Database connection
        conn = db_connection()
        cursor = conn.cursor()

        # Fetch latest total_units for accumulation
        cursor.execute(
            "SELECT total_units FROM transactions WHERE scheme_code = %s ORDER BY id DESC LIMIT 1",
            (scheme_code,)
        )
        result = cursor.fetchone()
        previous_total_units = Decimal(str(result[0])) if result and result[0] else Decimal(0)
        total_units = round(previous_total_units + units, 4)

        # Insert transaction
        cursor.execute(
            "INSERT INTO transactions (scheme_code, date, nav, amount, units, total_units) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (scheme_code, transaction_date.strftime('%Y-%m-%d'), latest_nav, amount, units, total_units)
        )

        conn.commit()
        cursor.close()
        conn.close()

        print(f"✅ Transaction Inserted: {scheme_code} | Total Units: {total_units}")

        return jsonify({
            "message": "Transaction added successfully.",
            "scheme_code": scheme_code,
            "transaction_date": transaction_date.strftime('%Y-%m-%d'),
            "latest_nav": float(latest_nav),
            "amount": float(amount),
            "units": float(units),
            "total_units": float(total_units)
        })

    except ValueError:
        return jsonify({"error": "Invalid data format. Please check the inputs."}), 400

    except Exception as e:
        print(f"Error Occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def get_latest_valid_nav(scheme_code, input_date):
    date = datetime.strptime(input_date, '%Y-%m-%d')
    while True:
        date_key = date.strftime('%d-%m')
        # Skip weekends and Indian holidays
        if date.weekday() not in (5, 6) and date_key not in INDIAN_HOLIDAYS:
            nav_data = mf.get_scheme_historical_nav(scheme_code)
            if nav_data and 'data' in nav_data:
                for entry in nav_data['data']:
                    entry_date = datetime.strptime(entry['date'], '%d-%m-%Y')
                    if entry_date <= date:
                        return Decimal(str(entry['nav']))
        date -= timedelta(days=1)  # Move to the previous day if invalid

@app.route('/sell_transaction', methods=['POST'])
def sell_transaction():
    try:
        data = request.json
        scheme_code = data.get('scheme_code')
        sell_amount = Decimal(str(data.get('amount', 0)))
        input_date = data.get('date')

        if sell_amount <= 0:
            return jsonify({"error": "Sell amount must be greater than zero."}), 400

        conn = db_connection()
        cursor = conn.cursor()

        # Fetch latest total units for the scheme
        cursor.execute(
            "SELECT total_units FROM transactions WHERE scheme_code = %s ORDER BY id DESC LIMIT 1", 
            (scheme_code,)
        )
        result = cursor.fetchone()

        if not result or result[0] is None:
            return jsonify({"error": "No units available for this scheme code."}), 400

        previous_total_units = Decimal(str(result[0]))
        
        # Get the latest valid NAV based on the input date
        latest_nav = get_latest_valid_nav(scheme_code, input_date)

        # Calculate units to sell
        units_to_sell = round(sell_amount / latest_nav, 4)
        if units_to_sell > previous_total_units:
            return jsonify({"error": "Insufficient units to sell."}), 400

        total_units = round(previous_total_units - units_to_sell, 4)

        # Insert sale transaction
        cursor.execute(
            "INSERT INTO transactions (scheme_code, date, nav, amount, units, total_units) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (scheme_code, input_date, latest_nav, -sell_amount, -units_to_sell, total_units)
        )

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "message": "Transaction (Sale) recorded successfully.",
            "scheme_code": scheme_code,
            "date": input_date,
            "latest_nav": float(latest_nav),
            "sell_amount": float(sell_amount),
            "units_sold": float(units_to_sell),
            "remaining_total_units": float(total_units)
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/get_average_nav/<scheme_code>', methods=['GET'])
def get_average_nav(scheme_code):
    try:
        conn = db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT SUM(amount), MAX(total_units) FROM transactions WHERE scheme_code = %s",
            (scheme_code,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        total_invested = Decimal(str(result[0])) if result[0] else Decimal(0)
        total_units = Decimal(str(result[1])) if result[1] else Decimal(0)

        average_nav = round(total_invested / total_units, 4) if total_units > 0 else Decimal(0)

        return jsonify({"scheme_code": scheme_code, "average_nav": float(average_nav)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_transactions/<scheme_code>')
def get_transactions(scheme_code):
    conn = db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT date, nav, amount, units, total_units FROM transactions WHERE scheme_code = %s ORDER BY date DESC",
        (scheme_code,)
    )
    transactions = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(transactions)



@app.route('/get_all_funds')
def get_all_funds():
    conn = db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM funds")
    funds = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(funds)

@app.route('/remove_fund', methods=['POST'])
def remove_fund():
    scheme_code = request.json.get('scheme_code')

    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM funds WHERE scheme_code = %s", (scheme_code,))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Fund removed successfully"})

@app.route('/get_nifty_value')
def get_nifty_value():
    nifty = yf.Ticker("^NSEBANK")
    nifty_data = nifty.history(period="1d")

    if nifty_data.empty:
        return jsonify({"error": "NIFTY 50 data unavailable"}), 500

    latest_price = nifty_data["Close"].iloc[-1]
    return jsonify({"nifty_value": latest_price})

if __name__ == '__main__':
    app.run(debug=True)