from flask import Blueprint, current_app, jsonify, render_template, request

from rag.chain import responder

chat_bp = Blueprint("chat", __name__)


@chat_bp.get("/")
def index():
    return render_template("index.html")


@chat_bp.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@chat_bp.post("/api/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    pergunta = (data.get("pergunta") or "").strip()

    if not pergunta:
        return jsonify({"erro": "A pergunta não pode ser vazia."}), 400

    try:
        resposta = responder(pergunta)
        return jsonify({"resposta": resposta})
    except Exception as exc:
        current_app.logger.exception("Falha ao processar /api/chat")
        return (
            jsonify(
                {
                    "erro": (
                        "Nao foi possivel processar a consulta no momento. "
                        f"Detalhe: {exc}"
                    )
                }
            ),
            503,
        )
