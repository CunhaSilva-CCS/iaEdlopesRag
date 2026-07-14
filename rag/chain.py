import os
import threading

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .loader import criar_retriever

_retriever = None
_cadeia = None
_init_lock = threading.Lock()
_init_state = "idle"
_init_error = ""


def _inicializar() -> None:
    global _retriever, _cadeia, _init_state, _init_error

    with _init_lock:
        if _init_state == "ready" and _retriever is not None and _cadeia is not None:
            return
        if _init_state == "building":
            return
        _init_state = "building"
        _init_error = ""

    def _job() -> None:
        global _retriever, _cadeia, _init_state, _init_error
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Variavel de ambiente OPENAI_API_KEY nao configurada. "
                    "Defina a chave no servico do Render."
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
        except Exception as exc:
            with _init_lock:
                _init_state = "error"
                _init_error = str(exc)

    threading.Thread(target=_job, daemon=True).start()


def responder(pergunta: str) -> str:
    _inicializar()

    with _init_lock:
        estado = _init_state
        erro = _init_error
        retriever = _retriever
        cadeia = _cadeia

    if estado != "ready" or retriever is None or cadeia is None:
        if estado == "error":
            raise RuntimeError(
                f"Falha na inicializacao da base de conhecimento: {erro}"
            )
        raise RuntimeError(
            "Base de conhecimento em preparacao. "
            "A indexacao dos PDFs esta em andamento, tente novamente em instantes."
        )

    trechos = retriever.invoke(pergunta)
    if not trechos:
        return "Não encontrei essa informação nos documentos disponíveis da Edlopes Transportes."
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)
    return cadeia.invoke({"query": pergunta, "contexto": contexto})
