from flask import Flask, request

app = Flask(__name__)

from filterLib.curse_detector import CurseDetector


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/text', methods=["POST"])
def check_text():

    curse = CurseDetector()
    body = request.json
    text = body['text']

    if curse.predict(text)[1] > 0.9:
        return 'true';

    return 'false'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
