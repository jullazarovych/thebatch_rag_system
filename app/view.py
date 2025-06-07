from flask import  render_template, current_app

@current_app.route('/')
def main():
    return render_template("base.html")
