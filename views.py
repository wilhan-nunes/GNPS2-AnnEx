import flask

blueprint = flask.Blueprint("ui", __name__)


@blueprint.route("/", methods=["GET"])
def render_homepage():
    return flask.render_template("homepage.html")


@blueprint.route("/dashinterface1", methods=["GET"])
@blueprint.route("/dashinterface1/", methods=["GET"])
def redirect_legacy_dashboard():
    query_string = flask.request.query_string.decode("utf-8").strip()
    redirect_target = "/dashinterface2/"
    if query_string:
        redirect_target = f"{redirect_target}?{query_string}"

    return flask.redirect(redirect_target, code=302)

#include health check
@blueprint.route("/health", methods=["GET"])
def health_check():
    return "OK", 200