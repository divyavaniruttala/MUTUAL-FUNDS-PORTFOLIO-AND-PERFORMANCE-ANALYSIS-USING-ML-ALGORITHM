import mysql.connector

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456789',
    'database': 'mutual_funds'
}

def db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            print("Database connected successfully!")
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def get_dict_cursor(connection):
    """Returns a cursor that fetches results as dictionaries."""
    return connection.cursor(dictionary=True)
