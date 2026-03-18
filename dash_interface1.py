"""Compatibility shim for the retired Dash interface 1 entry point."""

from dash_interface2 import dash_app


if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", port=5000, debug=True)
