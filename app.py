import streamlit as st
import numpy as np
import json
from PIL import Image

@st.cache_resource
def load_model():
    import ctypes, pathlib
    # Load TFLite model using numpy only
    with open('fashion_model.tflite', 'rb') as f:
        model_data = f.read()
    with open('classes.json') as f:
        class_names = json.load(f)
    return model_data, class_names

st.set_page_config(page_title="Fashion Classifier", page_icon="👗")
st.title("👗 Fashion Classifier")
st.write("Upload an image — AI will predict the clothing type!")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except:
        TFLITE_AVAILABLE = False

with open('classes.json') as f:
    class_names = json.load(f)

CLASS_ICONS = {'T-Shirt': '👕', 'Dress': '👗', 'Pants': '👖'}

if not TFLITE_AVAILABLE:
    st.error("Model runtime not available!")
else:
    if 'tflite_runtime' in str(type(tflite)):
        interpreter = tflite.Interpreter(model_path='fashion_model.tflite')
    else:
        interpreter = tflite.Interpreter(model_path='fashion_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns([1, 1.4], gap="large")
        with col1:
            st.image(img, use_column_width=True)
        with col2:
            with st.spinner("Predicting..."):
                img_arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                interpreter.set_tensor(input_details[0]['index'], img_arr)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])

            idx = str(np.argmax(pred))
            cls_name = class_names[idx]
            conf = pred[0][int(idx)] * 100
            icon = CLASS_ICONS.get(cls_name, '🧥')

            st.success(f"Prediction: {icon} **{cls_name}**")
            st.info(f"Confidence: {conf:.1f}%")

            if conf >= 90:
                st.success("✅ High confidence!")
            elif conf >= 70:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❌ Low confidence")

            st.write("### All Probabilities:")
            for i, name in class_names.items():
                prob = float(pred[0][int(i)])
                st.write(f"**{name}:** {prob*100:.1f}%")
                st.progress(prob)
