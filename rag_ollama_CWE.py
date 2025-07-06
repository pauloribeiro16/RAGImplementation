# rag_ollama_cwe_final_completo_v2.py

import os
import sys
import requests
import beaupy
import logging
import datetime
import time
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

# --- DEPENDÊNCIAS ---
# pip install llama-index beaupy llama-index-llms-ollama llama-index-embeddings-ollama sentence-transformers torch

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
    Document
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import NodeWithScore

# --- CONFIGURAÇÃO INICIAL DO LOGGING ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Silencia logs de HTTP

# --- CONFIGURAÇÕES E CONSTANTES ---
DOCS_DIRECTORY_PATH = "./docs"
PERSIST_DIR = "./storage_cwe_from_xml"
LOG_DIR_NAME = "evaluation_logs"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
RE_RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

# --- CASOS DE TESTE RELEVANTES PARA CWE ---
TEST_CASES = [
    { "category": "CWE Definition", "question": "What is CWE-1004?", "golden_answer": "CWE-1004 is named 'Sensitive Cookie Without 'HttpOnly' Flag'. It occurs when a web application uses a cookie for sensitive information but fails to set the HttpOnly flag, making it accessible to client-side scripts and vulnerable to theft via XSS." },
]

# --- CLASSE TestLogger (sem alterações) ---
class TestLogger:
    def __init__(self): self.log_filepath = None
    def initialize(self, model_name, run_mode):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_DIR_NAME); os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); safe_model_name = "".join(c for c in model_name if c.isalnum() or c in ('_')).replace(":", "_")
        self.log_filepath = os.path.join(log_dir, f"eval_{run_mode}_{safe_model_name}_{timestamp}.txt")
        header = ["="*60, "             CWE RAG EVALUATION LOG", "="*60, f"MODEL: {model_name}", f"RUN MODE: {run_mode}", f"DATE: {datetime.datetime.now().isoformat()}", "="*60]
        try:
            with open(self.log_filepath, 'w', encoding='utf-8') as f: f.write("\n".join(header) + "\n\n")
        except IOError as e: logging.error(f"Could not create log file: {e}"); self.log_filepath = None
    def log_test_case(self, test_case, no_rag_result, rag_result, test_index):
        if not self.log_filepath: return
        no_rag_prompt_system = no_rag_result['system_prompt']; no_rag_prompt_user = no_rag_result['user_prompt']; no_rag_response = no_rag_result['response']; no_rag_time = no_rag_result['duration']
        rag_full_prompt_for_log = rag_result['debug_prompt_for_log']; rag_response = rag_result['response']; rag_time = rag_result['duration']
        lines = [
            f"------------------------------------------------------------", f"TEST CASE {test_index}: {test_case['category']}", f"------------------------------------------------------------",
            "QUESTION:", f"{test_case['question']}\n", "GOLDEN ANSWER:", f"{test_case['golden_answer']}\n",
            f"--- NO-RAG RESPONSE (took {no_rag_time:.2f} seconds) ---\n",
            "  SYSTEM PROMPT:", f"  {no_rag_prompt_system}\n", "  USER PROMPT:", f"  {no_rag_prompt_user}\n", "  MODEL RESPONSE:", f"  {no_rag_response}\n",
            f"--- RAG RESPONSE (took {rag_time:.2f} seconds) ---\n",
            "  DEBUGGING INFO (RAW RETRIEVAL + PROMPT STRUCTURE):", f"{rag_full_prompt_for_log}\n", "  MODEL RESPONSE:", f"  {rag_response}\n", "="*60 ]
        try:
            with open(self.log_filepath, 'a', encoding='utf-8') as f: f.write("\n".join(lines) + "\n\n")
        except IOError as e: logging.error(f"Could not write to log file: {e}")
    def log_summary(self, timings):
        if not self.log_filepath: return
        rag_times = timings.get('rag', []); no_rag_times = timings.get('no_rag', [])
        total_time = sum(rag_times) + sum(no_rag_times); avg_rag = sum(rag_times) / len(rag_times) if rag_times else 0; avg_no_rag = sum(no_rag_times) / len(no_rag_times) if no_rag_times else 0
        summary = ["\n------------------------------------------------------------", "PERFORMANCE SUMMARY", "------------------------------------------------------------", f"Average RAG Response Time:       {avg_rag:.2f} seconds", f"Average No-RAG Response Time:    {avg_no_rag:.2f} seconds", f"Total Model Test Time:           {total_time:.2f} seconds", "------------------------------------------------------------"]
        try:
            with open(self.log_filepath, 'a', encoding='utf-8') as f: f.write("\n".join(summary))
        except IOError as e: logging.error(f"Could not write summary to log file: {e}")

# --- PARSER DE XML PERSONALIZADO PARA CWE (sem alterações) ---
def parse_cwe_xml(file_path: str) -> List[Document]:
    logging.info(f"Parsing CWE XML file: {file_path}")
    ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}
    try:
        tree = ET.parse(file_path); root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Failed to parse XML file {file_path}: {e}"); return []
    llama_documents = []
    for weakness_node in root.findall('.//cwe:Weakness', ns):
        cwe_id = weakness_node.get('ID', 'N/A'); cwe_name = weakness_node.get('Name', 'N/A')
        text_parts = []; metadata = {'source_file': os.path.basename(file_path), 'cwe_id': f"CWE-{cwe_id}", 'cwe_name': cwe_name, 'abstraction': weakness_node.get('Abstraction', 'N/A')}
        text_parts.append(f"CWE-{cwe_id}: {cwe_name}\n")
        def get_text(element, path):
            node = element.find(path, ns)
            return node.text.strip() if node is not None and node.text else ""
        text_parts.append(f"Description:\n{get_text(weakness_node, 'cwe:Description')}\n")
        extended_desc = get_text(weakness_node, 'cwe:Extended_Description')
        if extended_desc: text_parts.append(f"Extended Description:\n{extended_desc}\n")
        consequences = []; mitigations = []; examples = []
        for cons_node in weakness_node.findall('.//cwe:Consequence', ns):
            scope = get_text(cons_node, 'cwe:Scope'); impact = get_text(cons_node, 'cwe:Impact'); note = get_text(cons_node, 'cwe:Note')
            consequences.append(f"- Scope: {scope}, Impact: {impact}. Note: {note}")
        if consequences: text_parts.append("Common Consequences:\n" + "\n".join(consequences) + "\n")
        for mitig_node in weakness_node.findall('.//cwe:Potential_Mitigations/cwe:Mitigation', ns):
            phase = get_text(mitig_node, 'cwe:Phase'); desc = get_text(mitig_node, 'cwe:Description')
            mitigations.append(f"- (Phase: {phase}) {desc}")
        if mitigations: text_parts.append("Potential Mitigations:\n" + "\n".join(mitigations) + "\n")
        for ex_node in weakness_node.findall('.//cwe:Demonstrative_Example', ns):
            intro = get_text(ex_node, 'cwe:Intro_Text'); example_body = "\n".join(p.text.strip() for p in ex_node.findall('cwe:Body_Text', ns) if p.text)
            code_snippets = []
            for code_node in ex_node.findall('cwe:Example_Code', ns):
                code_type = code_node.get('Nature', 'Code'); code_lang = code_node.get('Language', ''); code = "".join(code_node.itertext()).strip()
                code_snippets.append(f"  {code_type} Example ({code_lang}):\n  <code>\n{code}\n  </code>")
            examples.append(f"Demonstrative Example:\n{intro}\n{example_body}\n" + "\n".join(code_snippets))
        if examples: text_parts.append("\n".join(examples) + "\n")
        full_text = "\n".join(text_parts); doc = Document(text=full_text, metadata=metadata)
        llama_documents.append(doc)
    logging.info(f"Successfully parsed {len(llama_documents)} weaknesses from XML.")
    return llama_documents

# --- LÓGICA DE INDEXAÇÃO E RETRIEVAL (COM CONTAGEM EXPLÍCITA) ---
def get_or_create_index(embed_model: OllamaEmbedding) -> VectorStoreIndex:
    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window", original_text_metadata_key="original_text")
    Settings.llm = Ollama(model="dummy"); Settings.embed_model = embed_model; Settings.node_parser = node_parser
    
    if os.path.exists(PERSIST_DIR):
        logging.info(f"Loading existing index from '{PERSIST_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR); index = load_index_from_storage(storage_context)
    else:
        logging.info("Index not found. Starting data processing pipeline.")
        if not os.path.exists(DOCS_DIRECTORY_PATH) or not any(fname.endswith('.xml') for fname in os.listdir(DOCS_DIRECTORY_PATH)):
             logging.error(f"Directory '{DOCS_DIRECTORY_PATH}' does not contain an XML file."); sys.exit(1)
        
        # PASSO 1: PARSING DO XML
        all_documents = []
        for xml_file in [f for f in os.listdir(DOCS_DIRECTORY_PATH) if f.endswith('.xml')]:
            all_documents.extend(parse_cwe_xml(os.path.join(DOCS_DIRECTORY_PATH, xml_file)))
        if not all_documents: logging.error("XML parsing resulted in zero documents."); sys.exit(1)
        
        # <-- MUDANÇA: DESMEMBRAMENTO DO PROCESSO -->
        # PASSO 2: PARSING DOS NÓS (SENTENÇAS)
        logging.info("XML parsing complete. Now parsing documents into sentence nodes for embedding...")
        nodes = Settings.node_parser.get_nodes_from_documents(all_documents, show_progress=True)
        
        # PASSO 3: INFORMAR O UTILIZADOR E GERAR EMBEDDINGS
        total_nodes = len(nodes)
        logging.info(f"Node parsing complete. A total of {total_nodes} sentence nodes will now be embedded.")
        logging.info("This is the main, one-time setup cost. The progress bar will now track the embedding of these nodes.")
        
        # A criação do índice agora é feita a partir dos nós pré-processados
        index = VectorStoreIndex(nodes, show_progress=True)
        
        logging.info(f"Embedding process complete. Persisting index to '{PERSIST_DIR}' for future fast reloads..."); index.storage_context.persist(persist_dir=PERSIST_DIR)

    logging.info("Index is ready.")
    return index

# --- RESTO DO SCRIPT (get_rag_response, perform_test_for_model, etc. SEM ALTERAÇÕES) ---
def get_rag_response(question: str, index: VectorStoreIndex, llm: Ollama, reranker: SentenceTransformerRerank) -> Dict:
    start_time = time.perf_counter()
    retriever = index.as_retriever(similarity_top_k=10); raw_retrieved_nodes = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(raw_retrieved_nodes, query_bundle=QueryBundle(question))
    postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window"); final_nodes_with_context = postprocessor.postprocess_nodes(reranked_nodes)
    llm_context_chunks = [node.get_content() for node in final_nodes_with_context]; llm_context_str = "\n\n---\n\n".join(llm_context_chunks)
    system_prompt = "You are a helpful and honest assistant specializing in cybersecurity weaknesses. Your primary goal is to answer the user's question accurately. First, carefully review the context provided below. If the context contains a direct and sufficient answer to the question, use that information to formulate your response. If the context is irrelevant, insufficient, or does not seem to contain the answer, state that you could not find a specific answer in the provided document, and then proceed to answer the question based on your own general knowledge. Always be clear about the source of your information (whether it's from the document or your general knowledge)."
    full_prompt_for_llm = (f"System: {system_prompt}\n\nContext:\n{llm_context_str}\n\nQuestion: {question}\n\nAnswer:")
    response = llm.complete(full_prompt_for_llm); duration = time.perf_counter() - start_time
    debug_raw_chunks = [f"--- RAW CHUNK {i+1} (Score: {n.score:.4f}) ---\n{n.get_content()}" for i, n in enumerate(raw_retrieved_nodes)]
    debug_prompt_for_log = (f"\n  [DEBUG] RAW RETRIEVED CHUNKS (Top 10):\n" + "\n".join(debug_raw_chunks) + f"\n\n  [DEBUG] FINAL CONTEXT SENT TO LLM (after re-ranking and context window):\n{llm_context_str}")
    return { "full_prompt_for_llm": full_prompt_for_llm, "debug_prompt_for_log": debug_prompt_for_log, "response": str(response).strip(), "duration": duration }
def get_no_rag_response(question: str, llm: Ollama) -> Dict:
    system_prompt = "You are an expert assistant on cybersecurity topics and the CWE framework. Answer the user's question based on your general knowledge."
    start_time = time.perf_counter()
    response = llm.complete(question, system_prompt=system_prompt); duration = time.perf_counter() - start_time
    return {"system_prompt": system_prompt, "user_prompt": question, "response": str(response).strip(), "duration": duration}
def perform_test_for_model(model_name: str, index: VectorStoreIndex, run_mode: str, reranker: SentenceTransformerRerank):
    print("\n" + "="*60); print(f"Executing Test Suite for Model: {model_name}"); print("="*60)
    llm = Ollama(model=model_name, base_url=OLLAMA_BASE_URL, request_timeout=300.0)
    logger = TestLogger(); logger.initialize(model_name, run_mode=run_mode)
    all_timings = {"rag": [], "no_rag": []}
    for i, test_case in enumerate(TEST_CASES):
        question = test_case["question"]
        print(f"--> Running test {i+1}/{len(TEST_CASES)}: {test_case['category']}")
        no_rag_result = get_no_rag_response(question, llm); all_timings["no_rag"].append(no_rag_result['duration'])
        rag_result = get_rag_response(question, index, llm, reranker); all_timings["rag"].append(rag_result['duration'])
        logger.log_test_case(test_case, no_rag_result, rag_result, test_index=i+1)
    logger.log_summary(all_timings); print(f"\nTest for model {model_name} complete. Log saved to: {logger.log_filepath}")
def list_ollama_models(exclude_model_prefix: str) -> List[str]:
    logging.info("Contacting Ollama...");
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags"); r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return sorted([m for m in models if not m.startswith(exclude_model_prefix)])
    except Exception as e: logging.error(f"Error contacting Ollama: {e}"); return []
def run_test_logic(models_to_test: List[str], embed_model: OllamaEmbedding, run_mode: str):
    if not models_to_test: logging.warning(f"No models found for run mode '{run_mode}'."); return
    logging.info("Initializing the Sentence Re-Ranker model..."); logging.info("On the first run, this may trigger a model download. Please wait.")
    reranker = SentenceTransformerRerank(model=RE_RANKER_MODEL, top_n=5)
    logging.info("Re-Ranker initialized successfully.")
    index = get_or_create_index(embed_model)
    for model_name in models_to_test:
        perform_test_for_model(model_name, index, run_mode=run_mode, reranker=reranker)
def run_all_models_test(embed_model: OllamaEmbedding):
    available_models = list_ollama_models(exclude_model_prefix=EMBEDDING_MODEL_NAME)
    print(f"\nFound {len(available_models)} models to test: {', '.join(available_models)}")
    run_test_logic(available_models, embed_model, run_mode="full_run")
def run_single_model_test(embed_model: OllamaEmbedding):
    available_models = list_ollama_models(exclude_model_prefix=EMBEDDING_MODEL_NAME)
    if not available_models: return
    print("\nPlease select a model to test:"); selected_model = beaupy.select(available_models, cursor="> ", cursor_style="cyan")
    if not selected_model: print("No model selected."); return
    run_test_logic([selected_model], embed_model, run_mode="single_run")
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