# 🫁 PneumoScan AI — Pneumonia Detection from Chest X-Rays

An end-to-end deep learning web application that detects pneumonia from chest X-ray images using a **5-model ensemble** with **97.3% accuracy**. The system provides visual explanations via Grad-CAM, showing exactly which lung regions influenced the diagnosis.

## 🚀 Live Demo  
[🔗 View Live App](https://chest-xray-pneumonia-ensemble-2rkuap9g8sc5meizrgfluy.streamlit.app/) 

## 🛠️ Tech Stack  
- **Engine:** Python 3.10+, PyTorch 2.x  
- **Deep Learning:** torchvision, timm (Custom CNN, ResNet, DenseNet, EfficientNet)  
- **Web Framework:** Streamlit  
- **Explainability:** Grad-CAM (custom implementation)  
- **Image Processing:** OpenCV, CLAHE  
- **Visuals:** Plotly, Matplotlib  

## 📊 Model Performance

The system combines five deep learning models using a weighted ensemble, achieving state‑of‑the‑art results on the Kaggle Chest X‑Ray dataset:

| Model               | Accuracy | F1 Score | AUC-ROC |
|---------------------|----------|----------|---------|
| Custom CNN          | 87.5%    | 87.5%    | 93.2%   |
| ResNet-18           | 92.3%    | 92.4%    | 96.5%   |
| ResNet-50           | 94.1%    | 94.1%    | 97.4%   |
| DenseNet-121        | 95.7%    | 95.6%    | 98.3%   |
| EfficientNet-B0     | 93.9%    | 93.8%    | 97.1%   |
| **Ensemble (weighted)** ⭐ | **97.3%** | **97.3%** | **99.1%** |

- **Optimization:** Pretrained models adapted for grayscale input, fine‑tuned on chest X‑rays.  
- **Reliability:** Real‑time confidence scores and Grad‑CAM heatmaps for every prediction.

## 💡 Features

- **Multi‑Model Ensemble:** Five independent deep learning models work together for maximum accuracy.  
- **Grad‑CAM Explainability:** Highlights the lung regions the AI focuses on — critical for clinical trust.  
- **CLAHE Preprocessing:** Enhances local contrast in X‑rays, improving feature visibility.  
- **Interactive Dashboard:** Upload an X‑ray, get an instant diagnosis, and explore model‑wise performance.  
- **Performance Analytics:** Compare individual models vs. ensemble via ROC curves, radar charts, and confusion matrix.  
- **Cyber‑Medical UI:** Clean dark‑mode interface with intuitive controls and visual feedback.

## 🧠 How It Works

1. **Image Upload & Preprocessing:**  
   Chest X‑ray is resized to 224×224, enhanced with CLAHE, and normalized.  

2. **Ensemble Inference:**  
   The image is passed through all five models. Their softmax probabilities are combined via a weighted average (weights optimized on validation data).  

3. **Diagnosis & Confidence:**  
   Final prediction (`NORMAL` or `PNEUMONIA`) is displayed alongside a confidence percentage.  

4. **Grad‑CAM Visualization:**  
   For the ensemble’s top contributor (e.g., DenseNet‑121), a heatmap is overlaid on the original X‑ray to show influential regions.  

5. **Analytics Dashboard:**  
   Explore detailed metrics and compare model performances.

## 📚 Dataset

**Chest X‑Ray Images (Pneumonia)** – [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- 5,216 training images (NORMAL + PNEUMONIA)  
- 624 test images  
- Augmentations: random crop, horizontal flip, rotation (±15°), color jitter

## 🔥 Grad‑CAM Explainability

Gradient‑weighted Class Activation Mapping provides visual insight into the model’s decision:

1. Forward pass through DenseNet‑121 → store last convolutional layer activations.  
2. Backward pass → compute gradients of the target class.  
3. Global average pooling of gradients → channel importance weights.  
4. Weighted combination of activation maps + ReLU → raw heatmap.  
5. Upsample to original size and blend (alpha=0.4) with input X‑ray.

The resulting overlay shows exactly which lung areas contributed most to the prediction.

## ⚕️ Disclaimer

**For research and educational purposes only.** Not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

## 🖥️ Local Development (Optional)

If you wish to run the app locally:

```bash
git clone https://github.com/HafsaIbrahim5/pneumoscan-ai
cd pneumoscan-ai
pip install -r requirements.txt
# Place your trained ensemble model (pneumonia_ensemble_full.pth) in the project root
streamlit run app.py
