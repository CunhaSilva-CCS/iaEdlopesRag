import os
import threading
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AuthenticationError

from .loader import buscar_contexto_rapido, criar_retriever

_retriever = None
_cadeia = None
_init_lock = threading.Lock()
_init_state = "idle"
_init_error = ""
_init_started_at = 0.0
_init_error_at = 0.0
_INIT_MAX_SECONDS = int(os.getenv("RAG_INIT_MAX_SECONDS", "240"))
_INIT_RETRY_SECONDS = int(os.getenv("RAG_INIT_RETRY_SECONDS", "45"))
_PREPARE_FALLBACK_SECONDS = int(os.getenv("RAG_PREPARE_FALLBACK_SECONDS", "45"))


def _normalizar_api_key(valor: str | None) -> str:
    if not valor:
        return ""

    chave = valor.strip().strip('"').strip("'")
    if chave.startswith("OPENAI_API_KEY="):
        chave = chave.split("=", 1)[1].strip().strip('"').strip("'")

    return chave


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
            api_key = _normalizar_api_key(os.getenv("OPENAI_API_KEY"))
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
        except AuthenticationError:
            with _init_lock:
                _init_state = "error"
                _init_error = (
                    "OPENAI_API_KEY invalida ou sem permissao para este projeto. "
                    "Atualize a chave no Render e reinicie o servico."
                )
                _init_error_at = time.monotonic()
                _init_started_at = 0.0
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

        if tempo_aguardo >= _PREPARE_FALLBACK_SECONDS:
            contexto_rapido = buscar_contexto_rapido(pergunta)
            if contexto_rapido:
                blocos = []
                for item in contexto_rapido:
                    blocos.append(
                        f"Fonte: {item['arquivo']}\nTrecho preliminar:\n{item['trecho']}"
                    )
                return (
                    "A base vetorial ainda esta em preparacao, mas localizei trechos preliminares nos PDFs:\n\n"
                    + "\n\n".join(blocos)
                )

            return (
                "A base vetorial ainda esta em preparacao e os primeiros PDFs analisados "
                "nao trouxeram trechos textuais aproveitaveis para esta pergunta. "
                "Tente novamente em 1-2 minutos."
            )

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
