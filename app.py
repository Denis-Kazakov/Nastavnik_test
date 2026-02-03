from typing import Literal
from pickle import load
import logging

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
import pandas as pd


logging.basicConfig(level=logging.ERROR)

class StudentQuery(BaseModel):
    user_id: str
    skill_id: str
    num_attempts: int = Field(..., gt=0)
    success_rate: float = Field(..., ge=0, le=1)
    last_correct: Literal[0, 1]
    learning_curve: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    knows_skill: bool
    confidence: float = Field(..., ge=0, le=1)


app = FastAPI(title="Nastavnik Knowledge Prediction")
logging.debug('App started')

_CLF = None
_SCALER = None

def load_models():
    global _CLF, _SCALER
    try:
        with open('model.pkl', "rb") as f:
            _CLF = load(f)
        logging.debug('Classification model loaded')
    except Exception as e:
        logging.error(f'Failed to load classifier with error {e}')

    try:
        with open('scaler.pkl', "rb") as f:
            _SCALER = load(f)
        logging.debug('Scaler loaded')
    except Exception as e:
        logging.error(f'Failed to load scaler with error {e}')


def get_cached_classifier():
    if _CLF is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    return _CLF


def get_cached_scaler():
    if _SCALER is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    return _SCALER


@app.on_event("startup")
def startup_event():
    load_models()


@app.post("/predict_knowledge", response_model=PredictionResponse)
async def predict_knowledge(
        query: StudentQuery,
        scaler = Depends(get_cached_scaler),
        clf = Depends(get_cached_classifier)
):
    """
    Предсказать, знает ли студент навык
    knows_skill: True если модель выдала > 0.5
    confidence: сырое предсказание модели (вероятность от 0 до 1)
    """
    try:
        x = pd.DataFrame(
            {
                'num_attempts': [query.num_attempts],
                'success_rate': [query.success_rate],
                'last_correct': [query.last_correct],
                'learning_curve': [query.learning_curve]
            }
        )

        x['num_attempts'] = scaler.transform(x[['num_attempts']])
        prediction = clf.predict_proba(x)[0][1]
        output = PredictionResponse(
            knows_skill=prediction > 0.5,
            confidence=prediction
        )
        return output
    except Exception as error:
        logging.error(f'Failed to predict skills with error {error}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error))


@app.get("/health")
def health_check():
    if _SCALER and _CLF:
        return {"status": "healthy"}
    else:
        logging.error('Failed to load models')
        return {"status": "unhealthy"}

