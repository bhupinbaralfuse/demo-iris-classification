from fastapi import FastAPI, Body
from typing import List
import pickle
import uvicorn

app = FastAPI()

# Load the model (assuming it's a scikit-learn model)
with open('model.pkl', 'rb') as file:
    MODEL = pickle.load(file)


@app.post("/api")
async def predict(data: List[float] = Body(...)):
    flower_data = data  # No preprocessing
    prediction = MODEL.predict([flower_data])[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000)
