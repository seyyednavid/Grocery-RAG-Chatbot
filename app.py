# app.py
from flask import Flask, render_template, request, session, jsonify
from rag_chain import ask_bot
import uuid
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-key")

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    session_id = session["session_id"]

    bot_reply = ask_bot(user_message, session_id)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
