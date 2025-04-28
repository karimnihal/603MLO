# lab9app/lab9app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

class Comment(BaseModel):
    reddit_comment: str

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("reddit_model_pipeline.joblib")
    print("✅  Model pipeline loaded!")

@app.get("/")
def root():
    return {"message": "This is a model for classifying Reddit comments"}

@app.post("/predict")
def predict(body: Comment):
    pred = model.predict([body.reddit_comment])[0]    # 0 or 1
    return {"prediction": int(pred)}
