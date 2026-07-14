# RAGEd 0.1

Aplicação Flask com RAG para consulta de documentos internos em PDF.

## Rodar localmente

1. Crie e ative um ambiente virtual.
   > Para manter compatibilidade com o deploy no Render, use Python 3.11.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Crie seu `.env` a partir do exemplo:
   ```bash
   cp .env.example .env
   ```
4. Preencha `OPENAI_API_KEY` no `.env`.
5. Inicie:
   ```bash
   python app.py
   ```

## Publicação no Render

Este projeto já está preparado para Render com:
- `render.yaml`
- `Procfile`
- `gunicorn` no `requirements.txt`

### Opção 1 (mais simples): Blueprint com `render.yaml`

1. Suba este projeto para um repositório no GitHub.
2. No Render, clique em **New +** → **Blueprint**.
3. Selecione o repositório.
4. Defina a variável de ambiente `OPENAI_API_KEY`.
5. Faça o deploy.

### Opção 2: Web Service manual

1. No Render, clique em **New +** → **Web Service**.
2. Conecte o repositório.
3. Use:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Defina `OPENAI_API_KEY`.
5. Faça o deploy.

## Observações

- O `.env` não deve ser enviado ao repositório.
- O índice vetorial (`.faiss_index/`) é cache local e é recriado quando necessário.
- O arquivo `.python-version` fixa o deploy em Python 3.11, que e compativel com a versao usada de `faiss-cpu` no Render.
# iaEdlopesRag
