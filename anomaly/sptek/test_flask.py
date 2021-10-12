import json
from flask import Flask
from flask import request
from flask import jsonify, Response


app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return Response(
                json.dumps(
                    {
                        "do_the_login": "failed"
                    },
                indent=4),
            mimetype='application/json',
            status=200
        )

    else:
        return "show_the_login_form"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5050', debug=True)


######
# execute flask
# $ python test_flask.py
######

