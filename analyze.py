from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize FastAPI app
app = FastAPI()

# Define ResumeAnalyzer class
class ResumeAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sid = SentimentIntensityAnalyzer()

    def keyword_analysis(self, resume_text, job_description_text, top_n=10):
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([job_description_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(vectors.sum(axis=0)).flatten()
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_keywords = [feature_names[i] for i in top_indices]

        resume_vectorizer = CountVectorizer(stop_words='english')
        resume_vectors = resume_vectorizer.fit_transform([resume_text])
        resume_feature_names = resume_vectorizer.get_feature_names_out()

        resume_keyword_counts = dict(zip(resume_feature_names, resume_vectors.sum(axis=0).A1))

        keyword_analysis_df = pd.DataFrame(top_keywords, columns=['Keyword'])
        keyword_analysis_df['Matches'] = keyword_analysis_df['Keyword'].map(resume_keyword_counts).fillna(0)

        return keyword_analysis_df.to_dict()

    def sentence_bert_similarity(self, resume_text, job_description_text):
        sentences = [resume_text, job_description_text]
        embeddings = self.model.encode(sentences)
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

    def readability_score_fk(self, cv_text, jd_text):
        return {
            "Flesch-Kincaid-cv": textstat.flesch_kincaid_grade(cv_text),
            "Flesch-Kincaid-jd": textstat.flesch_kincaid_grade(jd_text)
        }

    def readability_score_gf(self, cv_text, jd_text):
        return {
            "Gunning-Fog-cv": textstat.gunning_fog(cv_text),
            "Gunning-Fog-jd": textstat.gunning_fog(jd_text)
        }

    def sentiment_analysis(self, text):
        return self.sid.polarity_scores(text)

# Input data model
class ResumeData(BaseModel):
    resume_text: str
    job_description_text: str

# Initialize ResumeAnalyzer
analyzer = ResumeAnalyzer()

# API endpoint to analyze the resume and job description
@app.post("/analyze/")
def analyze_resume(data: ResumeData) -> Dict:
    resume_text = data.resume_text
    job_description_text = data.job_description_text

    try:
        # Perform all analyses
        similarity_score = analyzer.sentence_bert_similarity(resume_text, job_description_text)
        readability_fk = analyzer.readability_score_fk(resume_text, job_description_text)
        readability_gf = analyzer.readability_score_gf(resume_text, job_description_text)
        sentiment_analysis = analyzer.sentiment_analysis(resume_text)
        keyword_analysis = analyzer.keyword_analysis(resume_text, job_description_text)

        # Create response dictionary
        result = {
            "similarity_score": similarity_score,
            "readability_fk": readability_fk,
            "readability_gf": readability_gf,
            "sentiment_analysis": sentiment_analysis,
            "keyword_analysis": keyword_analysis
        }

        # Return JSON response
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
