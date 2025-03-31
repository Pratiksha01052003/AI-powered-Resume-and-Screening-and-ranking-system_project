import os
import PyPDF2
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def rank_resumes(resume_folder, job_description):
    resumes = []
    resume_names = []
    
    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf") or filename.endswith(".docx"):
            file_path = os.path.join(resume_folder, filename)
            text = extract_text(file_path)
            preprocessed_text = preprocess_text(text)
            resumes.append(preprocessed_text)
            resume_names.append(filename)
    
    resumes.append(preprocess_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(resumes)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    ranked_resumes = sorted(zip(resume_names, similarity_scores[0]), key=lambda x: x[1], reverse=True)
    
    print("Ranked Resumes:")
    for rank, (name, score) in enumerate(ranked_resumes, start=1):
        print(f"{rank}. {name} - Score: {score:.4f}")
    
    return ranked_resumes

# Example Usage
resume_folder = "resumes"  # Folder containing resumes
job_description = "Looking for a data scientist with experience in Python, machine learning, and NLP."
rank_resumes(resume_folder, job_description)
