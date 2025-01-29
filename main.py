from flask import Flask, render_template, request, jsonify
from controllers.chat_controller import get_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if msg:
        response = get_response(msg)
        return jsonify(response)
    return jsonify("Lo siento, no pude procesar tu solicitud.")

if __name__ == '__main__':
    app.run(debug=True)
