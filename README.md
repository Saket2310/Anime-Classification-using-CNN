🎌 Anime & Pokémon Classification Using CNN
📌 Project Overview
This project is about classifying images of Anime and Pokémon characters using a deep learning model. We use MobileNetV2, a pre-trained Convolutional Neural Network (CNN), and add extra layers to improve classification accuracy.

Using this model, we can automatically recognize different Anime and Pokémon characters based on images. This is useful for Anime fans, game developers, or anyone working with Anime image datasets.

🌟 Features
✔️ Uses MobileNetV2 – A lightweight and efficient CNN model
✔️ Data Preprocessing – Normalization, resizing, and shuffling of images
✔️ Transfer Learning – Improves accuracy with pre-trained features
✔️ Custom Fully Connected Layers – Helps in fine-tuning classification
✔️ Performance Evaluation – Generates accuracy scores & confusion matrix

📂 Dataset
We use a dataset containing Anime and Pokémon images, where each category (or class) has its own folder. The images are resized to 224x224 pixels to match the MobileNetV2 input format.

🛠 Installation & Setup
🔹 Step 1: Install Dependencies
Make sure you have Python installed. Then, install the required libraries using:
pip install tensorflow numpy opencv-python matplotlib scikit-learn
Alternatively, if you have a requirements.txt file, install all dependencies at once:
pip install -r requirements.txt

🔹 Step 2: Run the Jupyter Notebook
jupyter notebook Anime_Classification_using_CNN.ipynb
Follow the steps in the notebook to load data, preprocess it, and train the model.

🔧 How It Works
Data Preprocessing:
Load images from different categories (Anime & Pokémon).
Resize them to 224x224 pixels.
Normalize pixel values for better performance.
Model Setup:
Load MobileNetV2 (pre-trained on ImageNet).
Add extra Fully Connected (FC) layers to customize it for Anime/Pokémon classification.
Training & Evaluation:
Train the model using a dataset split into training & validation sets.
Evaluate the model using accuracy, loss, and confusion matrix.

📊 Model & Techniques
Feature	Details
Base Model	MobileNetV2
Input Image Size	224x224 pixels
Preprocessing	Normalization, Resizing
Training Framework	TensorFlow / Keras
Performance Metrics	Accuracy, Confusion Matrix

📈 Results
After training, the model outputs:
✅ Accuracy Report – Shows how well the model classifies images
✅ Confusion Matrix – Displays correct and incorrect classifications
✅ Predicted vs. Actual Labels – Helps in understanding model errors

💡 Future Improvements
🔹 Increase dataset size for better generalization
🔹 Experiment with other CNN architectures like ResNet or EfficientNet
🔹 Fine-tune hyperparameters to improve accuracy
🔹 Add a web interface to classify uploaded images

🤝 Contributing
🚀 Want to improve this project? Feel free to contribute!
