import os
import sys
import requests
import beaupy
import logging
import datetime
import time
from typing import List, Dict, Any

from llama_index.core import (
    Settings,  # <-- ESTA É A ADIÇÃO NECESSÁRIA
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    QueryBundle
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import NodeWithScore

# --- CONFIGURAÇÕES E CONSTANTES ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PDF_DIRECTORY_PATH = "./docs"
PERSIST_DIR = "./storage_gdpr_from_txt" # Novo diretório para o novo índice
LOG_DIR_NAME = "evaluation_logs"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
RE_RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


# --- GDPR TEST CASES (ENGLISH) ---
TEST_CASES = [
    { "category": "Core Definitions", "question": "What constitutes 'personal data' under GDPR?", "golden_answer": "Personal data is any information that relates to an identified or identifiable living individual. This includes obvious identifiers like a name or an ID number, as well as other data like location, IP address, or biometric data that can be used to identify a person." }
]

# --- TestLogger, get_no_rag_response (sem alterações) ---
# (O código para estas classes e funções é idêntico ao da versão anterior)
class TestLogger:
    def __init__(self): self.log_filepath = None
    def initialize(self, model_name, run_mode):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_DIR_NAME); os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('_')).replace(":", "_")
        self.log_filepath = os.path.join(log_dir, f"eval_{run_mode}_{safe_model_name}_{timestamp}.txt")
        header = ["="*60, "             GDPR RAG EVALUATION LOG", "="*60, f"MODEL: {model_name}", f"RUN MODE: {run_mode}", f"DATE: {datetime.datetime.now().isoformat()}", "="*60]
        try:
            with open(self.log_filepath, 'w', encoding='utf-8') as f: f.write("\n".join(header) + "\n\n")
        except IOError as e: logging.error(f"Could not create log file: {e}"); self.log_filepath = None

    def log_test_case(self, test_case, no_rag_result, rag_result, test_index):
        if not self.log_filepath: return
        no_rag_prompt_system = no_rag_result['system_prompt']; no_rag_prompt_user = no_rag_result['user_prompt']; no_rag_response = no_rag_result['response']; no_rag_time = no_rag_result['duration']
        rag_full_prompt = rag_result['full_prompt_for_llm']; rag_debug_info = rag_result['debug_prompt_for_log']; rag_response = rag_result['response']; rag_time = rag_result['duration']
        
        lines = [
            f"------------------------------------------------------------", f"TEST CASE {test_index}: {test_case['category']}", f"------------------------------------------------------------",
            "QUESTION:", f"{test_case['question']}\n",
            "GOLDEN ANSWER:", f"{test_case['golden_answer']}\n",
            f"--- NO-RAG RESPONSE (took {no_rag_time:.2f} seconds) ---\n",
            "  SYSTEM PROMPT:", f"  {no_rag_prompt_system}\n",
            "  USER PROMPT:", f"  {no_rag_prompt_user}\n",
            "  MODEL RESPONSE:", f"  {no_rag_response}\n",
            f"--- RAG RESPONSE (took {rag_time:.2f} seconds) ---\n",
            "  DEBUGGING INFO (RAW RETRIEVAL + PROMPT STRUCTURE):", f"{rag_debug_info}\n",
            "  MODEL RESPONSE:", f"  {rag_response}\n",
            "="*60
        ]
        try:
            with open(self.log_filepath, 'a', encoding='utf-8') as f: f.write("\n".join(lines) + "\n\n")
        except IOError as e: logging.error(f"Could not write to log file: {e}")

    def log_summary(self, timings):
        if not self.log_filepath: return
        rag_times = timings.get('rag', []); no_rag_times = timings.get('no_rag', [])
        total_time = sum(rag_times) + sum(no_rag_times)
        avg_rag = sum(rag_times) / len(rag_times) if rag_times else 0; avg_no_rag = sum(no_rag_times) / len(no_rag_times) if no_rag_times else 0
        summary = ["\n------------------------------------------------------------", "PERFORMANCE SUMMARY", "------------------------------------------------------------", f"Average RAG Response Time:       {avg_rag:.2f} seconds", f"Average No-RAG Response Time:    {avg_no_rag:.2f} seconds", f"Total Model Test Time:           {total_time:.2f} seconds", "------------------------------------------------------------"]
        try:
            with open(self.log_filepath, 'a', encoding='utf-8') as f: f.write("\n".join(summary))
        except IOError as e: logging.error(f"Could not write summary to log file: {e}")

# --- NOVA ABORDAGEM DE INDEXAÇÃO E RETRIEVAL ---

def get_or_create_index(embed_model: OllamaEmbedding) -> VectorStoreIndex:
    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window", original_text_metadata_key="original_text")
    Settings.node_parser = node_parser
    Settings.embed_model = embed_model
    # O LLM não é usado na indexação, mas é exigido pelas Settings
    Settings.llm = Ollama(model="dummy") 

    if os.path.exists(PERSIST_DIR):
        logging.info(f"Loading existing index from '{PERSIST_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        logging.info(f"Index not found. Creating new index from .txt file...")
        if not os.path.exists(PDF_DIRECTORY_PATH) or not any(fname.endswith('.txt') for fname in os.listdir(PDF_DIRECTORY_PATH)):
             logging.error(f"Directory '{PDF_DIRECTORY_PATH}' does not contain a .txt file.")
             sys.exit(1)
        
        # <-- MUDANÇA: Lê apenas ficheiros de texto.
        reader = SimpleDirectoryReader(PDF_DIRECTORY_PATH, required_exts=[".txt"])
        documents = reader.load_data()
        
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        logging.info(f"Persisting index to '{PERSIST_DIR}'...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    logging.info("Index is ready.")
    return index

def get_rag_response(question: str, index: VectorStoreIndex, llm: Ollama) -> Dict:
    start_time = time.perf_counter()

    # <-- MUDANÇA: Recupera 10 chunks para máxima transparência
    retriever = index.as_retriever(similarity_top_k=10)
    raw_retrieved_nodes = retriever.retrieve(question)

    # Re-rank para obter os 5 melhores para usar no prompt
    reranker = SentenceTransformerRerank(model=RE_RANKER_MODEL, top_n=5)
    reranked_nodes = reranker.postprocess_nodes(raw_retrieved_nodes, query_bundle=QueryBundle(question))
    
    # Expande o contexto APENAS para os nós re-classificados que serão usados
    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
    final_nodes_with_context = postprocessor.postprocess_nodes(reranked_nodes)

    # Constrói a secção de contexto para o prompt final do LLM
    llm_context_chunks = [node.get_content() for node in final_nodes_with_context]
    llm_context_str = "\n\n---\n\n".join(llm_context_chunks)
    
    system_prompt = "You are a helpful and honest assistant... (o mesmo prompt resiliente de antes)"
    
    full_prompt_for_llm = (
        f"System: {system_prompt}\n\n"
        f"Context:\n{llm_context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = llm.complete(full_prompt_for_llm)
    duration = time.perf_counter() - start_time
    
    # Constrói a secção de debug para o log
    debug_raw_chunks = [f"--- RAW CHUNK {i+1} (Score: {n.score:.4f}) ---\n{n.get_content()}" for i, n in enumerate(raw_retrieved_nodes)]
    debug_prompt_for_log = (
        f"\n  [DEBUG] RAW RETRIEVED CHUNKS (Top 10):\n" + "\n".join(debug_raw_chunks) +
        f"\n\n  [DEBUG] FINAL CONTEXT SENT TO LLM (after re-ranking and context window):\n{llm_context_str}"
    )

    return {
        "full_prompt_for_llm": full_prompt_for_llm,
        "debug_prompt_for_log": debug_prompt_for_log,
        "response": str(response).strip(),
        "duration": duration
    }

def get_no_rag_response(question: str, llm: Ollama) -> Dict:
    system_prompt = "You are an expert assistant on legal and data privacy topics. Answer the user's question based on your general knowledge."
    start_time = time.perf_counter()
    response = llm.complete(question, system_prompt=system_prompt)
    duration = time.perf_counter() - start_time
    return {"system_prompt": system_prompt, "user_prompt": question, "response": str(response).strip(), "duration": duration}

def perform_test_for_model(model_name: str, index: VectorStoreIndex, run_mode: str):
    print("\n" + "="*60); print(f"Executing Test Suite for Model: {model_name}"); print("="*60)
    llm = Ollama(model=model_name, base_url=OLLAMA_BASE_URL, request_timeout=300.0)
    logger = TestLogger(); logger.initialize(model_name, run_mode=run_mode)
    all_timings = {"rag": [], "no_rag": []}
    for i, test_case in enumerate(TEST_CASES):
        question = test_case["question"]
        print(f"--> Running test {i+1}/{len(TEST_CASES)}: {test_case['category']}")
        no_rag_result = get_no_rag_response(question, llm)
        all_timings["no_rag"].append(no_rag_result['duration'])
        rag_result = get_rag_response(question, index, llm)
        all_timings["rag"].append(rag_result['duration'])
        logger.log_test_case(test_case, no_rag_result, rag_result, test_index=i+1)
    logger.log_summary(all_timings)
    print(f"\nTest for model {model_name} complete. Log saved to: {logger.log_filepath}")

# --- Funções de Main e Wrappers (sem alterações significativas) ---
def list_ollama_models(exclude_model_prefix: str) -> List[str]:
    logging.info("Contacting Ollama...")
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags"); r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return sorted([m for m in models if not m.startswith(exclude_model_prefix)])
    except Exception as e: logging.error(f"Error contacting Ollama: {e}"); return []
def run_all_models_test(embed_model: OllamaEmbedding):
    available_models = list_ollama_models(exclude_model_prefix=EMBEDDING_MODEL_NAME)
    if not available_models: logging.warning("No models found to test."); return
    print(f"\nFound {len(available_models)} models to test: {', '.join(available_models)}")
    index = get_or_create_index(embed_model)
    for model_name in available_models: perform_test_for_model(model_name, index, run_mode="full_run")
def run_single_model_test(embed_model: OllamaEmbedding):
    available_models = list_ollama_models(exclude_model_prefix=EMBEDDING_MODEL_NAME)
    if not available_models: return
    print("\nPlease select a model to test:"); selected_model = beaupy.select(available_models, cursor="> ", cursor_style="cyan")
    if not selected_model: print("No model selected."); return
    index = get_or_create_index(embed_model)
    perform_test_for_model(selected_model, index, run_mode="single_run")
def main():
    script_start_time = time.perf_counter()
    modes = ["Automated Test (All Models)", "Automated Test (Single Model)"]
    print("Please select the run mode:"); mode = beaupy.select(modes, cursor="> ", cursor_style="cyan")
    if not mode: print("No mode selected."); sys.exit(0)
    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    if mode == "Automated Test (All Models)": run_all_models_test(embed_model)
    elif mode == "Automated Test (Single Model)": run_single_model_test(embed_model)
    print(f"\nTotal script execution time: {time.perf_counter() - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()