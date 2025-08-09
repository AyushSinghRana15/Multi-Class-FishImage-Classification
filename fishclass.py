import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os

# -- CONFIG --
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

FISH_DESCRIPTIONS = {
    'animal fish': "A general category for various types of fish.",
    'animal fish bass': "Bass are popular freshwater game fish.",
    'fish sea_food black_sea_sprat': "Black sea sprat is a small forage fish important in marine ecosystems.",
    'fish sea_food gilt_head_bream': "Gilt-head bream is a prized fish in Mediterranean cuisine.",
    'fish sea_food hourse_mackerel': "Horse mackerel is known for its distinct taste and is widely consumed.",
    'fish sea_food red_mullet': "Red mullet is a flavorful Mediterranean fish often used in gourmet dishes.",
    'fish sea_food red_sea_bream': "Red sea bream is a popular fish in East Asian seafood dishes.",
    'fish sea_food sea_bass': "Sea bass is a common name for various species prized by anglers and chefs.",
    'fish sea_food shrimp': "Shrimp are crustaceans widely harvested for human consumption.",
    'fish sea_food striped_red_mullet': "Striped red mullet is a variant of red mullet with distinctive stripes.",
    'fish sea_food trout': "Trout are freshwater fish related to salmon, popular in sport fishing."
}

SAMPLE_IMAGES_DIR = 'sample_fish_images'  # Place sample images here with filenames named after class or descriptive names

# -- LOAD MODEL --
@st.cache_resource
def load_selected_model(model_path):
    return load_model(model_path)

def preprocess_image(img: PIL.Image.Image):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img, model):
    processed = preprocess_image(img)
    probs = model.predict(processed)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    return top3_idx, probs[top3_idx]

def plot_confidence(probs):
    st.subheader("🔍 Confidence Scores by Class")
    fig, ax = plt.subplots()
    bars = ax.barh(CLASS_NAMES, probs, color='skyblue')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, i, f'{probs[i]*100:.1f}%', va='center')
    st.pyplot(fig)

# -- STREAMLIT UI --
st.set_page_config(page_title="🐟 Fish Classifier", layout='centered')
st.title("🐟 Fish Species Classifier")

model_option = st.selectbox("Choose a model to use:", [
    'best_model.h5',
    'best_vgg16.h5',
    'best_resnet50.h5',
    'best_mobilenet.h5',
    'best_inceptionv3.h5',
    'best_efficientnetb0.h5'
])
model = load_selected_model(model_option)

# Sidebar for sample image selection
st.sidebar.header("Try a Sample Image")
sample_files = []
if os.path.exists(SAMPLE_IMAGES_DIR):
    sample_files = sorted(os.listdir(SAMPLE_IMAGES_DIR))
selected_sample = st.sidebar.selectbox("Select a sample fish image", ["None"] + sample_files)

uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

input_image = None
if selected_sample != "None":
    # Load sample image from directory
    input_image = PIL.Image.open(os.path.join(SAMPLE_IMAGES_DIR, selected_sample)).convert('RGB')
elif uploaded_file is not None:
    input_image = PIL.Image.open(uploaded_file).convert('RGB')

if input_image:
    st.image(input_image, caption='📷 Input Image', use_column_width=True)

    if st.button("🔮 Predict Species"):
        with st.spinner("Analyzing image..."):
            top3_indices, top3_probs = predict(input_image, model)

        st.success(f"🎯 **Top 3 Predictions:**")
        for idx, prob in zip(top3_indices, top3_probs):
            class_name = CLASS_NAMES[idx]
            description = FISH_DESCRIPTIONS.get(class_name, "Description not available.")
            st.markdown(f"**{class_name}**: {prob:.2%} confidence")
            with st.expander(f"More about {class_name}"):
                st.write(description)

        # Optionally show confidence for all classes
        all_probs = np.zeros(len(CLASS_NAMES))
        all_probs[top3_indices] = top3_probs
        plot_confidence(model.predict(preprocess_image(input_image))[0])

else:
    st.info("Please upload an image or select a sample to get started.")
