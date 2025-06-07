from flask import Flask

def create_app(config_name="default"):
    app = Flask(__name__)

    with app.app_context():
        from . import view
    return app  