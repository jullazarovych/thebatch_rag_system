from flask import Blueprint, render_template, request
from app.services.search import handle_query

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            return handle_query(query)
    return render_template("base.html")
