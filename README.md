# RAGImplementation Project

## Project Purpose

The primary purpose of the `RAGImplementation` project is to serve as a **robust and transparent testing environment** for evaluating, comparing, and optimizing the components of a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of creating a "black-box" RAG application, this project focuses on dissecting each stage of the process to answer fundamental questions about its effectiveness. The evaluation is conducted using language models running locally via **Ollama**, ensuring a controlled and private testing environment.

## Core Objectives

The objectives of this evaluation framework are:

1.  **Compare Performance (RAG vs. No-RAG):**
    *   To quantitatively measure the impact of RAG on response quality.
    *   For each query, generate one response from the LLM without any context (relying solely on its internal knowledge) and a second response enriched with retrieved context, enabling a direct, side-by-side comparison.

2.  **Evaluate the Impact of Data Parsing:**
    *   To demonstrate how the quality of the input data source radically affects the entire pipeline.
    *   The project is designed to test different parsing strategies, from reading structured text files to detailed XML parsing, proving that data extraction and formatting are critical steps for success.

3.  **Measure the Effectiveness of Advanced Retrieval Techniques:**
    *   To go beyond simple vector search.
    *   To implement and evaluate more sophisticated retrieval strategies, such as **Sentence Window Retrieval** (to find precise sentences and expand their context) and **Re-ranking** (to use a smarter, secondary filter to refine initial search results).

4.  **Quantify Performance and Ensure Transparency:**
    *   To generate detailed, plain-text logs for every run.
    *   These logs provide full transparency into the process by recording:
        *   The exact system and user prompts sent to the LLM.
        *   For the RAG system, the exact text chunks that were retrieved, along with their scores.
        *   The response time (latency) for each model call, enabling performance analysis.

## Pipeline Architecture

The project's workflow is designed to be explicit and modular:

1.  **Custom Ingestion and Parsing:** The data source (e.g., a CWE XML or TXT file) is processed by a custom parser that understands the document's structure, extracting relevant text and metadata for each logical entry.
2.  **Indexing (Sentence-Window):** Instead of indexing large paragraphs, the system splits documents into individual sentences for vector search while preserving the context of neighboring sentences in the metadata.
3.  **Retrieval and Re-ranking:** A two-stage process finds the information:
    *   **Fast Retrieval:** A vector similarity search retrieves an initial set of candidate sentences (e.g., top 10).
    *   **Smart Filtering:** A re-ranking model (cross-encoder) then analyzes these candidates alongside the original query to re-sort them with much higher precision, selecting the best-fit results (e.g., top 5).
4.  **Resilient Context-Aware Generation:** A robust system prompt is constructed, instructing the LLM to use the provided context as its primary source but giving it the ability to fall back on its general knowledge if the context is irrelevant, thereby increasing the system's honesty and resilience.
