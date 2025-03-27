from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import base64
import os

from gradcam_utils import (
    load_model_and_layer,
    preprocess_image,
    generate_gradcam,
    extract_vit_feature
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(image)
        vit_vector = extract_vit_feature(image)

        results = {}
        confidences = {}
        models_dir = "models"

        for filename in os.listdir(models_dir):
            if filename.endswith(".keras"):
                condition = filename.replace("_model.keras", "").replace(".keras", "")
                model_path = os.path.join(models_dir, filename)

                model, last_conv_layer = load_model_and_layer(model_path)
                print(f"✅ Reloaded model: {filename} | Using layer: {last_conv_layer}")

                if len(model.inputs) == 1:
                    prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
                else:
                    prediction = model.predict({
                        "cnn_input": np.expand_dims(img_array, axis=0),
                        "vit_input": np.expand_dims(vit_vector, axis=0)
                    })[0][0]

                confidence = float(prediction)
                results[condition] = {
                    "label": "Positive" if confidence > 0.5 else "Negative",
                    "confidence": confidence
                }
                confidences[condition] = confidence

        gradcams = []
        if confidences:
            max_conf = max(confidences.values())
            for filename in os.listdir(models_dir):
                if filename.endswith(".keras"):
                    condition = filename.replace("_model.keras", "").replace(".keras", "")
                    if confidences[condition] == max_conf:
                        model_path = os.path.join(models_dir, filename)
                        model, last_conv_layer = load_model_and_layer(model_path)
                        if last_conv_layer:
                            try:
                                heatmap = generate_gradcam(model, img_array, last_conv_layer)
                                superimposed_img = np.uint8(heatmap)
                                img_pil = Image.fromarray(superimposed_img)
                                buffered = io.BytesIO()
                                img_pil.save(buffered, format="PNG")
                                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                                gradcams.append({"condition": condition, "gradcam": img_base64})
                            except Exception as e:
                                print(f"⚠️ Grad-CAM failed for {condition}: {str(e)}")
                        else:
                            print(f"⚠️ Skipping Grad-CAM for {condition}: No valid Conv2D layer.")

            results["gradcam_matches"] = gradcams if gradcams else []

        return JSONResponse(content=results)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)