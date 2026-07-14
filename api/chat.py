from flask import Blueprint, current_app, jsonify, render_template, request

from rag.chain import responder
from rag.loader import listar_documentos_disponiveis

chat_bp = Blueprint("chat", __name__)


@chat_bp.get("/")
def index():
    return render_template("index.html")


@chat_bp.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@chat_bp.get("/api/documentos")
def documentos():
    try:
        return jsonify({"documentos": listar_documentos_disponiveis()})
    except Exception as exc:
        current_app.logger.exception("Falha ao listar documentos")
        return jsonify({"erro": f"Falha ao listar documentos: {exc}"}), 500


@chat_bp.post("/api/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    pergunta = (data.get("pergunta") or "").strip()
    documentos_raw = data.get("documentos") or []
    documentos = [
        doc.strip() for doc in documentos_raw if isinstance(doc, str) and doc.strip()
    ]

    if not pergunta:
        return jsonify({"erro": "A pergunta não pode ser vazia."}), 400

    try:
        resposta = responder(pergunta, documentos=documentos)
        return jsonify({"resposta": resposta})
    except Exception as exc:
        erro_texto = str(exc)

        if "Base de conhecimento em preparacao" in erro_texto:
            return (
                jsonify(
                    {
                        "status": "preparando",
                        "erro": erro_texto,
                    }
                ),
                202,
            )

        if "Falha na inicializacao da base de conhecimento" in erro_texto:
            return jsonify({"status": "erro_inicializacao", "erro": erro_texto}), 503

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
