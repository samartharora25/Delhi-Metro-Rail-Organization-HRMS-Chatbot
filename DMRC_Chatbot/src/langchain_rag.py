import os
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None


class HRDocumentRAG:
    def __init__(
        self,
        chroma_db_path: str,
        collection_name: str = "hr_documents",
        pdf_path: Optional[str] = None,
        chunks_file: Optional[str] = None,
    ) -> None:
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.pdf_path = pdf_path
        self.chunks_file = chunks_file

        self._vectorstore: Optional[Chroma] = None
        self._retriever = None
        self._llm = None

    def build_rag_system(
        self,
        force_rebuild: bool = False,
        groq_model: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        os.makedirs(self.chroma_db_path, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if not force_rebuild:
            try:
                vs = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.chroma_db_path,
                    embedding_function=embeddings,
                )
                if vs._collection.count() > 0:  # type: ignore[attr-defined]
                    self._vectorstore = vs
                else:
                    self._vectorstore = None
            except Exception:
                self._vectorstore = None

        if self._vectorstore is None:
            docs = self._load_documents()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = splitter.split_documents(docs)

            self._vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=self.chroma_db_path,
            )
            try:
                self._vectorstore.persist()
            except Exception:
                pass

        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and ChatGroq is not None:
            self._llm = ChatGroq(model=groq_model, temperature=temperature, max_tokens=max_tokens)
        else:
            self._llm = None

    def _load_documents(self):
        if self.chunks_file:
            raise NotImplementedError("chunks_file is not supported in this build")

        if not self.pdf_path:
            raise ValueError("pdf_path is required")
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        loader = PyMuPDFLoader(self.pdf_path)
        return loader.load()

    def query(self, question: str, return_source_docs: bool = True) -> Any:
        if self._retriever is None:
            raise RuntimeError("RAG system not built. Call build_rag_system() first.")

        docs = self._retriever.invoke(question)

        if self._llm is None:
            if return_source_docs:
                return {"answer": "", "source_documents": docs}
            return docs

        context = "\n\n".join((d.page_content or "") for d in docs)
        prompt = (
            "You are an HR policy assistant. Answer using only the provided context. "
            "If the answer is not present, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        res = self._llm.invoke(prompt)
        answer = getattr(res, "content", None) or str(res)

        if return_source_docs:
            return {"answer": answer, "source_documents": docs}
        return answer

    def get_stats(self) -> Dict[str, Any]:
        if self._vectorstore is None:
            return {"ready": False}
        try:
            count = self._vectorstore._collection.count()  # type: ignore[attr-defined]
        except Exception:
            count = None
        return {
            "ready": True,
            "collection_name": self.collection_name,
            "persist_directory": self.chroma_db_path,
            "doc_count": count,
            "pdf_path": self.pdf_path,
        }
