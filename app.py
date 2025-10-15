import streamlit as st
import pandas as pd
import re
from unidecode import unidecode
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import docx
from io import BytesIO

# ---------- helper functions ----------
def clean_text(s):
    if not s: return ''
    s = unidecode(str(s))
    s = s.replace('\r',' ').replace('\n',' ')
    s = re.sub(r'[\u2022•\*·]+', ', ', s)
    s = re.sub(r'[^ -~]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s*,\s*', ', ', s)
    return s

def extract_text_from_pdf(file_bytes):
    text = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or '')
    return ' '.join(text)

def extract_text_from_docx(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return ' '.join(paragraphs)

def extract_uploaded_file_text(uploaded_file):
    if uploaded_file is None:
        return ''
    content = uploaded_file.read()
    fname = uploaded_file.name.lower()
    if fname.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif fname.endswith('.docx'):
        return extract_text_from_docx(content)
    elif fname.endswith('.txt'):
        return content.decode('utf-8', errors='ignore')
    else:
        # fallback to try text decode
        try:
            return content.decode('utf-8', errors='ignore')
        except:
            return ''

def normalize_skills_field_string(x, master_skill_list):
    # quick search of known skills in text
    t = x.lower()
    return {s for s in master_skill_list if s in t}

def jaccard(a,b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

# ---------- load jobs and model (cached) ----------
@st.cache_data
def load_jobs(path='jobs.csv'):
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # create combined_text column if not present
    jd_col = 'Job Description' if 'Job Description' in df.columns else df.columns[1]
    df['combined_text'] = df[jd_col].fillna('').apply(clean_text)
    return df

@st.cache_resource
def load_model_and_job_embs(job_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_embs = model.encode(job_texts, convert_to_tensor=True, show_progress_bar=False)
    return model, job_embs

jobs = load_jobs('jobs.csv')
job_texts = jobs['combined_text'].tolist()
model, job_embs = load_model_and_job_embs(job_texts)

# define master skill list (extend as needed)
master_skill_list = [
    'python','pandas','numpy','scikit-learn','sklearn','matplotlib','seaborn',
    'sql','power bi','tableau','excel','r','tensorflow','keras',
    'machine learning','deep learning','nlp','spark','aws','git',
    'javascript','java','c++','django','flask','react','node'
]

# ---------- UI ----------
st.title("Resume → Job Matcher")
st.write("Upload a resume (PDF / DOCX / TXT). The app will compute semantic similarity to job descriptions and show top matches.")

uploaded = st.file_uploader("Upload resume", type=['pdf','docx','txt'])
k = st.slider("Top k matches to show", 1, 10, 5)
alpha = st.slider("Semantic weight (alpha)", 0.0, 1.0, 0.75)

if uploaded:
    raw_text = extract_uploaded_file_text(uploaded)
    cleaned = clean_text(raw_text)
    st.subheader("Preview (first 1000 chars)")
    st.code(cleaned[:1000])

    # candidate embedding
    q_emb = model.encode(cleaned, convert_to_tensor=True)
    cosine_scores = util.cos_sim(q_emb, job_embs)[0]  # shape (n_jobs,)
    # get top k
    vals, idxs = torch.topk(cosine_scores, k=min(k, cosine_scores.shape[0]))
    rows = []
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        job_row = jobs.iloc[idx]
        sem_score = float(score)
        # skill overlap
        candidate_skills = normalize_skills_field_string(cleaned, master_skill_list)
        job_skills = set(job_row.get('combined_text','').lower())
        # better: compute job_skills using master list
        job_skills = normalize_skills_field_string(job_row.get('combined_text',''), master_skill_list)
        skill_overlap = jaccard(candidate_skills, job_skills)
        final = alpha * sem_score + (1-alpha) * skill_overlap
        rows.append({
            'job_title': job_row.get('Job Title', ''),
            'semantic_score': round(sem_score, 4),
            'skill_overlap': round(skill_overlap, 4),
            'final_pct': round(final*100,2),
            'job_description': job_row.get('Job Description', job_row.get('combined_text',''))
        })

    # display results
    st.subheader("Top matches")
    for r in rows:
        st.markdown(f"### {r['job_title']} — **{r['final_pct']}%**")
        st.write(f"Semantic: {r['semantic_score']}, Skill overlap: {r['skill_overlap']}")
        with st.expander("Show job description"):
            st.write(r['job_description'])

