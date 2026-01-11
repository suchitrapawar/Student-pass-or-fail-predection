import gradio as gr
import numpy as np
from sklearn.linear_model import LogisticRegression

# ----------------------------------
# Dummy training data (simulated)
# ----------------------------------
# Features: [internal_marks, external_marks, attendance]
X = np.array([
    [45, 48, 90], [40, 42, 85], [38, 40, 80],   # Pass
    [30, 35, 75], [28, 32, 70], [25, 30, 65],   # Borderline
    [20, 25, 60], [18, 22, 55], [15, 20, 50],   # Fail
    [10, 15, 40], [12, 18, 45], [8, 12, 35]
])

y = np.array([1,1,1,1,1,1,0,0,0,0,0,0])  # 1 = Pass, 0 = Fail

# ----------------------------------
# Train model ONCE
# ----------------------------------
model = LogisticRegression()
model.fit(X, y)

# ----------------------------------
# Prediction logic
# ----------------------------------
def predict_pass_fail(internal, external, attendance):
    features = np.array([[internal, external, attendance]])
    probability = model.predict_proba(features)[0][1]

    result = "PASS" if probability >= 0.5 else "FAIL"

    explanation = {
        "Pass Probability": round(probability * 100, 2),
        "Fail Probability": round((1 - probability) * 100, 2),
        "Decision": result
    }

    return result, explanation

# ----------------------------------
# Gradio UI
# ----------------------------------
interface = gr.Interface(
    fn=predict_pass_fail,
    inputs=[
        gr.Slider(0, 50, step=1, label="Internal Marks"),
        gr.Slider(0, 50, step=1, label="External Marks"),
        gr.Slider(0, 100, step=1, label="Attendance (%)")
    ],
    outputs=[
        gr.Label(label="Result"),
        gr.JSON(label="Prediction Details")
    ],
    title="Student Pass / Fail Prediction System",
    description="Predicts student performance using Logistic Regression"
)

# ----------------------------------
# Entry point
# ----------------------------------
interface.launch()
