from app import app

# This import is necessary so Flask registers the active Dash app.
import dash_interface2

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9092, debug=True)
