import subprocess
import sys

# Installation manuelle de rembg si nécessaire
try:
    from rembg import remove
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rembg"])
    from rembg import remove

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io
import numpy as np
import cv2

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Télécharge automatiquement le modèle si non disponible

st.title("Remove BG & Walmart Logos")

option = st.sidebar.selectbox("Choisir une fonction :", ("Remove Background", "Remove Walmart"))

if option == "Remove Background":
    st.header("Remove Background")
    uploaded_file = st.file_uploader("Télécharge une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image originale", use_column_width=True)
        
        # Supprime l'arrière-plan
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        output = remove(img_bytes)
        result_image = Image.open(io.BytesIO(output))
        st.image(result_image, caption="Image sans arrière-plan", use_column_width=True)
        
        # Téléchargement
        result_buffer = io.BytesIO()
        result_image.save(result_buffer, format="PNG")
        result_buffer.seek(0)
        st.download_button("Télécharger l'image", data=result_buffer, file_name="sans_arriere_plan.png", mime="image/png")

elif option == "Remove Walmart":
    st.header("Remove Walmart Logos")
    uploaded_file = st.file_uploader("Télécharge une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image originale", use_column_width=True)
        
        # Convertir l'image en format OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Détection avec YOLOv8
        results = model(image_cv)
        detected_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                if confidence > 0.5:  # Filtrer les détections pertinentes
                    detected_boxes.append((x1, y1, x2, y2))
        
        # Dessiner un masque sur les zones détectées
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        for box in detected_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 0))
        
        st.image(result_image, caption="Image sans logos Walmart", use_column_width=True)
        
        # Téléchargement
        result_buffer = io.BytesIO()
        result_image.save(result_buffer, format="PNG")
        result_buffer.seek(0)
        st.download_button("Télécharger l'image", data=result_buffer, file_name="sans_logo_walmart.png", mime="image/png")
