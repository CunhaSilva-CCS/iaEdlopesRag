import os
import threading
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .loader import criar_retriever

_retriever = None
_cadeia = None
_init_lock = threading.Lock()
_init_state = "idle"
_init_error = ""
_init_started_at = 0.0
_init_error_at = 0.0
_INIT_MAX_SECONDS = int(os.getenv("RAG_INIT_MAX_SECONDS", "900"))
_INIT_RETRY_SECONDS = int(os.getenv("RAG_INIT_RETRY_SECONDS", "120"))


def _inicializar() -> None:
    global _retriever, _cadeia, _init_state, _init_error, _init_started_at
    global _init_error_at

    agora = time.monotonic()

    with _init_lock:
        if _init_state == "ready" and _retriever is not None and _cadeia is not None:
            return

        if _init_state == "error":
            if _init_error_at and (agora - _init_error_at) < _INIT_RETRY_SECONDS:
                return
            _init_state = "idle"
            _init_error = ""

        if _init_state == "building":
            if _init_started_at and (agora - _init_started_at) > _INIT_MAX_SECONDS:
                _init_state = "error"
                _init_error = (
                    "Tempo limite na preparacao da base documental. "
                    "A indexacao excedeu o tempo esperado."
                )
                _init_error_at = agora
            return

        _init_state = "building"
        _init_error = ""
        _init_started_at = agora

    def _job() -> None:
        global _retriever, _cadeia, _init_state, _init_error, _init_started_at
        global _init_error_at
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Variavel de ambiente OPENAI_API_KEY nao configurada. "
                    "Defina a chave no servico do Render."
                )
            if not api_key.startswith("sk-"):
                raise RuntimeError(
                    "OPENAI_API_KEY com formato invalido. "
                    "A chave deve iniciar com 'sk-'."
                )

            modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=api_key)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Você é o assistente da Edlopes Transportes. "
                        "Responda exclusivamente com base no contexto fornecido dos documentos internos. "
                        "Se a informação não estiver no contexto, diga explicitamente que não encontrou "
                        "o conteúdo nos documentos disponíveis. "
                        "Use linguagem institucional, objetiva e técnica, com foco em segurança, conformidade e operação.",
                    ),
                    ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:"),
                ]
            )

            retriever = criar_retriever()
            cadeia = prompt | modelo | StrOutputParser()

            with _init_lock:
                _retriever = retriever
                _cadeia = cadeia
                _init_state = "ready"
                _init_error = ""
                _init_started_at = 0.0
                _init_error_at = 0.0
        except Exception as exc:
            with _init_lock:
                _init_state = "error"
                _init_error = str(exc)
                _init_error_at = time.monotonic()
                _init_started_at = 0.0

    threading.Thread(target=_job, daemon=True).start()


def preaquecer_base() -> None:
    _inicializar()


def responder(pergunta: str) -> str:
    _inicializar()

    agora = time.monotonic()

    with _init_lock:
        estado = _init_state
        erro = _init_error
        retriever = _retriever
        cadeia = _cadeia
        started_at = _init_started_at

    if estado != "ready" or retriever is None or cadeia is None:
        if estado == "error":
            raise RuntimeError(
                f"Falha na inicializacao da base de conhecimento: {erro}"
            )

        tempo_aguardo = int(agora - started_at) if started_at else 0
        raise RuntimeError(
            "Base de conhecimento em preparacao. "
            "A indexacao dos PDFs esta em andamento, tente novamente em instantes. "
            f"Tempo de preparo: {tempo_aguardo}s."
        )

    trechos = retriever.invoke(pergunta)
    if not trechos:
        return "Não encontrei essa informação nos documentos disponíveis da Edlopes Transportes."
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)
    return cadeia.invoke({"query": pergunta, "contexto": contexto})
