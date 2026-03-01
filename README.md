

# 🩺 HealthSenseAI

**AI Assistant for Public Health Awareness & Early Risk Guidance**

---

## 🏷️ At a Glance

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3.1%208B-orange)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-2E77BB.svg)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow.svg)
![RAG](https://img.shields.io/badge/Architecture-RAG-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📸 Application Screenshot

<img width="1877" height="910" alt="HealthsenseAI" src="https://github.com/user-attachments/assets/18d3f41a-cc27-4fdd-8375-03db3d6c8223" />


---

## 1. Executive Summary

**HealthSenseAI** is a multilingual, guideline-aware public health assistant built using **Generative AI + Retrieval-Augmented Generation (RAG)**.

It is designed to:

* Help citizens understand **symptoms, risks, prevention, and screenings** using credible public health content.
* Reduce the burden on hospitals and clinics for **non-emergency informational queries**.
* Demonstrate how **LLMs, vector search, and guardrails** can be combined to deliver safe, explainable AI in healthcare settings.

The solution retrieves content from uploaded **WHO / CDC / MoHFW** public-health guidelines, indexes them with FAISS, and responds using **Groq’s high-performance Llama 3.1 models**, wrapped in a clean Streamlit UI.

> **Important:** HealthSenseAI is an educational tool. It does **not** provide diagnosis, prescriptions, or clinical decision support.

---

## 2. Business Problem

Healthcare systems worldwide face similar challenges:

* ❌ Patients search on Google and receive **inconsistent, unverified information**
* ❌ Hospitals and helplines are overwhelmed with **non‑urgent questions**
* ❌ Public health guidelines are available but **hard to navigate**
* ❌ Reliable content is often **available only in English**

**Organizations affected:**

* Public hospitals and clinics
* Government health departments
* NGOs and community health programs
* Health insurers and wellness platforms
* Telemedicine providers

---

## 3. Solution Overview

HealthSenseAI provides:

* ✅ AI assistant grounded in trusted guideline documents
* ✅ Multilingual support (English, Hindi, Marathi)
* ✅ Retrieval‑Augmented Generation using FAISS
* ✅ Fast inference via Groq Llama models
* ✅ Streamlit‑based user interface
* ✅ Guardrails for safe healthcare responses

---

## 4. Architecture

![HealthSenseAI Architecture](assets/architecture_teal_white.png)

**Pipeline:**

1. User asks a health question
2. Query is converted into embeddings
3. FAISS retrieves relevant guideline chunks
4. Context sent to Groq LLM
5. LLM generates grounded response
6. Answer displayed in Streamlit UI

---

## 5. Project Structure

```
HealthSenseAI/
│
├── app.py
├── rag_pipeline.py
├── utils.py
├── requirements.txt
│
├── configs/
│   ├── configs.py
│   └── model_config.yaml
│
├── data/
│   └── raw/
│
├── vectorstore/
├── logs/
├── screenshots/
└── assets/
```

---

## 6. Installation

```bash
git clone https://github.com/YOUR_USERNAME/HealthSenseAI.git
cd HealthSenseAI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create `.env`

```
GROQ_API_KEY=your_api_key_here
```

---

## 7. Run Application

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## 8. Add Knowledge Base

Place PDFs in:

```
data/raw/
```

System automatically builds FAISS index.

---

## 9. Tech Stack

* Python
* Streamlit
* Groq LLM
* FAISS
* HuggingFace Embeddings
* Guardrails AI

---

## 10. Author

**Dr. Pankaj Mahure**
Public Health Professional | AI Developer

---

⭐ Star this repository if you find it useful
