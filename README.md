# Aquanex Sovereign

**Aquanex Sovereign** is an offline-first, air-gapped agentic RAG and engineering audit workbench built for industrial mechanical integrity—specifically **API 570 piping inspection audits**. Developed for the Smart India Hackathon (SIH), it allows refinery and plant engineering teams to query Piping and Instrumentation Diagrams (P&IDs), calculate corrosion rates, and determine remaining safe operating life across a local intranet without sending proprietary asset data to external cloud APIs.

---

## Why Aquanex Sovereign?
In heavy industrial and critical infrastructure facilities, maintenance logs, wall-thickness measurements, and P&ID schematics are strictly confidential and often reside inside air-gapped networks with zero internet access. Standard cloud-dependent AI tools fail or violate security policies in these environments. 

Aquanex Sovereign solves this by hosting the entire retrieval and reasoning pipeline locally on a single edge workstation while making the workbench accessible to any authorized computer on the facility's Local Area Network (LAN).

---

## Core Architecture & Features

* **Strict Zero-WAN / Air-Gapped Execution:**  
  Engineered to function reliably when Wi-Fi and external WAN links are completely disabled. The runtime enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` alongside numerical IPv4 loopback sockets (`127.0.0.1`) so local model loading never stalls on external model-hub checks or offline DNS resolution failures.

* **Edge GPU-Optimized Inference (NVIDIA RTX 3050):**  
  * **LLM Reasoning:** Powered locally by `qwen2.5:3b` via Ollama, pinned in VRAM (`keep_alive`) for fast, deterministic responses.
  * **CPU-Isolated Embeddings:** Document and query vectorization runs using `all-MiniLM-L6-v2` strictly on the CPU (`CUDA_VISIBLE_DEVICES=""` for the backend worker), reserving 100% of the RTX 3050's 6 GB VRAM for token generation and preventing Out-Of-Memory (OOM) crashes.

* **LAN-Wide Multi-Client Access:**  
  While the AI model remains locked to the host workstation's GPU, the FastAPI backend and React/Vite frontend bind across the local network interface (`0.0.0.0`). Any engineer's laptop or terminal on the same intranet can open the workbench in a web browser to run live audits simultaneously.

* **Deterministic API 570 Engineering Guardrails:**  
  Combines multi-step context extraction (identifying line tags, nominal vs. measured wall thicknesses, and service conditions) with explicit API 570 formulas:
  $$\text{Remaining Life (years)} = \frac{t_{\text{actual}} - t_{\text{required}}}{\text{Corrosion Rate}}$$
  Automatically flags lines where actual measured thickness ($t_{\text{actual}}$) approaches or drops below minimum required thickness ($t_{\text{required}}$), triggering immediate shutdown or pressure-derating alerts.

---

## System Workflow
1. **Document & Line Ingestion:** Engineering notes, thickness logs, and P&ID metadata are chunked and embedded locally on the CPU.
2. **Agentic Context Extraction:** When an engineer queries a line tag (e.g., `CDU-2-04-150-A1A`), the backend extracts baseline design parameters and current inspection readings.
3. **Sovereign Inference:** The structured context is routed to the local Ollama instance over `127.0.0.1:11434`, generating an auditable compliance assessment directly on the RTX 3050.
4. **Intranet Delivery:** Results stream in real time to any connected client browser on the local network.

---

## Tech Stack
* **Frontend:** React, TypeScript, Vite
* **Backend:** Python, FastAPI, Uvicorn, HTTPX
* **Local AI Engine:** Ollama (`qwen2.5:3b`)
* **Embeddings & RAG:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Deployment Environment:** Ubuntu / WSL2 on NVIDIA GeForce RTX 3050
