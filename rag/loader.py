import io
import json
import logging
import os
from contextlib import redirect_stderr
from hashlib import sha256
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

_RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "8"))
_FETCH_K = int(os.getenv("RAG_FETCH_K", "24"))
_SEARCH_TYPE = os.getenv("RAG_SEARCH_TYPE", "mmr")
_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
_EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DOCS_DIR = _PROJECT_DIR / "documentos"
_INDEX_DIR = _PROJECT_DIR / ".faiss_index"
_INDEX_NAME = "edlopes_docs"
_MANIFEST_PATH = _INDEX_DIR / "manifest.json"

logging.getLogger("pypdf").setLevel(logging.ERROR)
_LOGGER = logging.getLogger(__name__)


def _carregar_paginas_pdf(arquivo: Path):
    # Primeiro tenta PyPDF (mais leve). Se vier vazio, tenta fallback com PyMuPDF.
    stderr_buffer = io.StringIO()
    with redirect_stderr(stderr_buffer):
        carregados = PyPDFLoader(str(arquivo)).load()

    stderr_output = stderr_buffer.getvalue().strip()
    if stderr_output:
        _LOGGER.warning(
            "PDF com avisos estruturais (%s): %s",
            arquivo,
            stderr_output.splitlines()[0],
        )

    tem_texto = any((doc.page_content or "").strip() for doc in carregados)
    if tem_texto:
        return carregados

    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        fallback = PyMuPDFLoader(str(arquivo)).load()
        if any((doc.page_content or "").strip() for doc in fallback):
            _LOGGER.info("Fallback PyMuPDF aplicado com sucesso para %s", arquivo)
            return fallback
    except Exception as exc:
        _LOGGER.warning("Fallback PyMuPDF indisponivel para %s: %s", arquivo, exc)

    return carregados


def _obter_api_key_openai() -> SecretStr:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variavel de ambiente OPENAI_API_KEY nao configurada. "
            "Defina a chave no ambiente do Render."
        )
    return SecretStr(api_key)


def _listar_arquivos_pdf(base_dir: Path) -> list[Path]:
    arquivos = sorted(
        [
            arquivo
            for arquivo in base_dir.rglob("*")
            if arquivo.is_file() and arquivo.suffix.lower() == ".pdf"
        ]
    )
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo PDF encontrado em {base_dir}. "
            "Adicione documentos para indexação do assistente."
        )
    return arquivos


def _snapshot_arquivos_pdf(arquivos_pdf: list[Path]) -> list[dict[str, str | int]]:
    snapshot = []
    for arquivo in arquivos_pdf:
        stat = arquivo.stat()
        snapshot.append(
            {
                "path": str(arquivo.relative_to(_PROJECT_DIR)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def _assinatura_snapshot(snapshot: list[dict[str, str | int]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _manifest_atualizado(snapshot: list[dict[str, str | int]]) -> bool:
    if not _MANIFEST_PATH.exists():
        return False

    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return (
        manifest.get("snapshot_hash") == _assinatura_snapshot(snapshot)
        and manifest.get("chunk_size") == _CHUNK_SIZE
        and manifest.get("chunk_overlap") == _CHUNK_OVERLAP
    )


def _carregar_ou_criar_indice(
    embeddings: OpenAIEmbeddings, arquivos_pdf: list[Path]
) -> FAISS:
    snapshot = _snapshot_arquivos_pdf(arquivos_pdf)
    index_file = _INDEX_DIR / f"{_INDEX_NAME}.faiss"
    pkl_file = _INDEX_DIR / f"{_INDEX_NAME}.pkl"

    if _manifest_atualizado(snapshot) and index_file.exists() and pkl_file.exists():
        return FAISS.load_local(
            folder_path=str(_INDEX_DIR),
            embeddings=embeddings,
            index_name=_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )

    indice = None
    total_pedacos = 0
    arquivos_processados = 0
    arquivos_com_falha = 0
    for arquivo in arquivos_pdf:
        try:
            carregados = _carregar_paginas_pdf(arquivo)

            for documento in carregados:
                documento.metadata["source_file"] = arquivo.name
                documento.metadata["source_group"] = arquivo.parent.name
                documento.metadata["source_relpath"] = str(
                    arquivo.relative_to(_PROJECT_DIR)
                )

            pedacos = splitter.split_documents(carregados)
            if not pedacos:
                arquivos_com_falha += 1
                _LOGGER.warning("PDF sem conteudo aproveitavel: %s", arquivo)
                continue

            total_pedacos += len(pedacos)
            arquivos_processados += 1

            if indice is None:
                indice = FAISS.from_documents(pedacos, embeddings)
            else:
                indice.add_documents(pedacos)
        except Exception as exc:
            arquivos_com_falha += 1
            _LOGGER.warning("Falha ao processar PDF %s: %s", arquivo, exc)
            continue

    if indice is None:
        raise RuntimeError(
            "Nenhum trecho foi gerado a partir dos PDFs disponiveis para indexacao. "
            "Verifique se os arquivos PDF estao validos e legiveis."
        )

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    indice.save_local(folder_path=str(_INDEX_DIR), index_name=_INDEX_NAME)
    _MANIFEST_PATH.write_text(
        json.dumps(
            {
                "snapshot_hash": _assinatura_snapshot(snapshot),
                "document_count": len(arquivos_pdf),
                "processed_document_count": arquivos_processados,
                "failed_document_count": arquivos_com_falha,
                "chunk_count": total_pedacos,
                "chunk_size": _CHUNK_SIZE,
                "chunk_overlap": _CHUNK_OVERLAP,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return indice


def criar_retriever():
    embeddings = OpenAIEmbeddings(
        api_key=_obter_api_key_openai(),
        chunk_size=_EMBEDDING_BATCH_SIZE,
    )
    arquivos_pdf = _listar_arquivos_pdf(_DOCS_DIR)
    indice = _carregar_ou_criar_indice(embeddings, arquivos_pdf)

    search_kwargs = {"k": _RETRIEVAL_K}
    if _SEARCH_TYPE == "mmr":
        search_kwargs["fetch_k"] = _FETCH_K

    return indice.as_retriever(search_type=_SEARCH_TYPE, search_kwargs=search_kwargs)
