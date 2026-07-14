from flask import Blueprint, jsonify, render_template, request

from rag.chain import responder

chat_bp = Blueprint("chat", __name__)


@chat_bp.get("/")
def index():
    return render_template("index.html")


@chat_bp.post("/api/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    pergunta = (data.get("pergunta") or "").strip()

    if not pergunta:
        return jsonify({"erro": "A pergunta não pode ser vazia."}), 400

    resposta = responder(pergunta)
    return jsonify({"resposta": resposta})
