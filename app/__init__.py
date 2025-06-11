from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

def create_app(config_name="default"):
    app = Flask(__name__, template_folder='templates') 
    
    with app.app_context():
        from . import view
        app.register_blueprint(view.main_bp)
    
    return app  