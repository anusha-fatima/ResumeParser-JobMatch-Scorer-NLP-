import streamlit as st
import pandas as pd
import os
import re
from unidecode import unidecode
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import docx
from io import BytesIO

def clean_text(s):
    if not s: return ''
    s = unidecode(str(s))
    s = s.replace('\r',' ').replace('\n',' ')
    s = re.sub(r'[\u2022•\*·]+', ', ', s)
    s = re.sub(r'[^ -~]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s*,\s*', ', ', s)
    return s

def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or '')
    return ' '.join(text)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return ' '.join(paragraphs)

def extract_resume_text(path):
    path = path.lower()
    if path.endswith('.pdf'):
        return extract_text_from_pdf(path)
    elif path.endswith('.docx'):
        return extract_text_from_docx(path)
    elif path.endswith('.txt'):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        return ''

def normalize_skills_field_string(x, master_skill_list):
    t = x.lower()
    return {s for s in master_skill_list if s in t}

def jaccard(a,b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


master_skill_list = [
    'python','pandas','numpy','scikit-learn','sklearn','matplotlib','seaborn',
    'sql','power bi','tableau','excel','r','tensorflow','keras',
    'machine learning','deep learning','nlp','spark','aws','git',
    'javascript','java','c++','django','flask','react','node'
]

st.title("Resume → Job Match Scorer (Batch Mode)")
st.write("This app compares **multiple resumes** against a **single job description** and computes match scores.")

job_desc = st.text_area("Paste or type the job description here:")

resume_folder = st.text_input("Path to Resume Folder:", "resumes")
alpha = st.slider("Semantic weight (α)", 0.0, 1.0, 0.75)

if st.button("Evaluate All Resumes"):
    if not job_desc.strip():
        st.warning("Please enter a job description first.")
    elif not os.path.exists(resume_folder):
        st.error(f"Folder '{resume_folder}' not found.")
    else:
        resumes = [os.path.join(resume_folder, f) for f in os.listdir(resume_folder)
                   if f.lower().endswith(('.pdf','.docx','.txt'))]
        if not resumes:
            st.warning("No resumes found in the folder.")
        else:
            job_text = clean_text(job_desc)
            job_emb = model.encode(job_text, convert_to_tensor=True)

            results = []
            for path in resumes:
                resume_name = os.path.basename(path)
                raw_text = extract_resume_text(path)
                cleaned = clean_text(raw_text)
                resume_emb = model.encode(cleaned, convert_to_tensor=True)
                sem_score = float(util.cos_sim(resume_emb, job_emb)[0][0])

                
                resume_skills = normalize_skills_field_string(cleaned, master_skill_list)
                job_skills = normalize_skills_field_string(job_text, master_skill_list)
                skill_overlap = jaccard(resume_skills, job_skills)

                final = alpha * sem_score + (1-alpha) * skill_overlap
                results.append({
                    "Resume": resume_name,
                    "Semantic": round(sem_score, 4),
                    "Skill Overlap": round(skill_overlap, 4),
                    "Final Match %": round(final * 100, 2)
                })

            df = pd.DataFrame(results).sort_values("Final Match %", ascending=False).reset_index(drop=True)
            st.subheader("Match Results")
            st.dataframe(df)

            st.bar_chart(df.set_index("Resume")["Final Match %"])


