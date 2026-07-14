import os
from hashlib import sha256
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "8"))
_FETCH_K = int(os.getenv("RAG_FETCH_K", "24"))
_SEARCH_TYPE = os.getenv("RAG_SEARCH_TYPE", "mmr")
_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DOCS_DIR = _PROJECT_DIR / "documentos"
_INDEX_DIR = _PROJECT_DIR / ".faiss_index"
_INDEX_NAME = "edlopes_docs"
_MANIFEST_PATH = _INDEX_DIR / "manifest.json"


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

    documentos = []
    for arquivo in arquivos_pdf:
        carregados = PyPDFLoader(str(arquivo)).load()
        for documento in carregados:
            documento.metadata["source_file"] = arquivo.name
            documento.metadata["source_group"] = arquivo.parent.name
            documento.metadata["source_relpath"] = str(arquivo.relative_to(_PROJECT_DIR))
        documentos.extend(carregados)

    pedacos = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    ).split_documents(documentos)
    indice = FAISS.from_documents(pedacos, embeddings)

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    indice.save_local(folder_path=str(_INDEX_DIR), index_name=_INDEX_NAME)
    _MANIFEST_PATH.write_text(
        json.dumps(
            {
                "snapshot_hash": _assinatura_snapshot(snapshot),
                "document_count": len(arquivos_pdf),
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
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    arquivos_pdf = _listar_arquivos_pdf(_DOCS_DIR)
    indice = _carregar_ou_criar_indice(embeddings, arquivos_pdf)

    search_kwargs = {"k": _RETRIEVAL_K}
    if _SEARCH_TYPE == "mmr":
        search_kwargs["fetch_k"] = _FETCH_K

    return indice.as_retriever(search_type=_SEARCH_TYPE, search_kwargs=search_kwargs)
