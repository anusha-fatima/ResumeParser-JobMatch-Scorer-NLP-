## ResumeParser-JobMatch-Scorer-NLP

An NLP-based web application that analyzes resumes and matches them to job descriptions using **semantic similarity** and **skill overlap**. Built with **Python**, **Sentence Transformers**, and **Streamlit**.

---

## Project Overview

This project automatically evaluates resumes against a given job description and provides a **match score** that helps recruiters or job seekers understand how closely a resume fits a particular role.  
It combines **semantic understanding (NLP)** with **skill-based comparison** for more accurate results.

---

## Features

-  Select or enter a job description
-  Computes:
  - **Semantic Similarity** (using Sentence Transformers)
  - **Skill Overlap** (using Jaccard similarity)
- Final score combines both metrics:  
  \[
  \text{Final Score} = (\alpha \times \text{Semantic}) + (1 - \alpha) \times \text{SkillOverlap}
  \]
- [Streamlit App](https://me7dpsqgazwjmynbd5twgm.streamlit.app/)


---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – UI framework  
- **SentenceTransformers** – Semantic embeddings  
- **PyTorch** – Backend for NLP model  
- **Pandas** – Data handling  
- **pdfplumber / python-docx** – Resume parsing  
- **GitHub + Colab + ngrok** – Deployment and testing  

---

## How It Works

- Resume Parsing → Extracts text from uploaded resumes.

- Cleaning & Preprocessing → Removes noise and special characters.

- Semantic Matching → Measures meaning similarity using all-MiniLM-L6-v2.

- Skill Overlap → Uses Jaccard similarity to compare candidate vs job skills.

- Final Scoring → Weighted formula combining both metrics.
