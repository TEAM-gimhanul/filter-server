from flask import Flask, request, jsonify

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

    result = curse.predict(text)
    returnArr = []
    cnt = 0

    for i in result:
        if i[1] > 0.9:
            returnArr.append('true')
            cnt += 1
            continue

        returnArr.append('false')
        cnt += 1

    return jsonify(
        data=returnArr
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8085', debug=True)
