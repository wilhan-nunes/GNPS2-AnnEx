import os
from flask import Flask
from flask_caching import Cache

import views

APP_ROOT = os.path.dirname(os.path.realpath(__file__))
CACHE_TYPE = "filesystem"
CACHE_DIR = os.path.join(APP_ROOT, "temp", "flask-cache")
CACHE_DEFAULT_TIMEOUT = 86400
CACHE_THRESHOLD = 10000


class CustomFlask(Flask):
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        block_start_string='(%',
        block_end_string='%)',
        variable_start_string='((',
        variable_end_string='))',
        comment_start_string='(#',
        comment_end_string='#)'))

os.makedirs(CACHE_DIR, exist_ok=True)

app = CustomFlask(__name__)
app.config.from_object(__name__)
cache = Cache(app)
app.register_blueprint(views.blueprint)
