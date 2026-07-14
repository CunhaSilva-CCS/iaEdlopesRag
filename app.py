import os

from dotenv import load_dotenv

load_dotenv()

from flask import Flask

from api.chat import chat_bp

app = Flask(__name__)
app.register_blueprint(chat_bp)

if __name__ == "__main__":
    porta = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=porta, debug=True)
