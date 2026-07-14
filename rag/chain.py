import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .loader import criar_retriever

_retriever = None
_cadeia = None


def _inicializar() -> None:
    global _retriever, _cadeia
    if _retriever is not None:
        return

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

    _retriever = criar_retriever()
    _cadeia = prompt | modelo | StrOutputParser()


def responder(pergunta: str) -> str:
    _inicializar()
    trechos = _retriever.invoke(pergunta)
    if not trechos:
        return "Não encontrei essa informação nos documentos disponíveis da Edlopes Transportes."
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)
    return _cadeia.invoke({"query": pergunta, "contexto": contexto})
