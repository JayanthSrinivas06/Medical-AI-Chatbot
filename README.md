# 🩺 Medical AI Chatbot for First‑Aid Assistance

An AI‑powered **medical chatbot** designed to provide **basic first‑aid guidance** by classifying user queries into predefined medical intents using a **dual‑input deep learning architecture**. The model combines **contextual understanding (LSTM)** with **keyword emphasis (Bag‑of‑Words)** to achieve robust intent classification for healthcare‑related conversations.

---

## 📌 Project Overview

The core idea of this project is to improve intent classification in medical chatbots by leveraging two complementary NLP representations:

* **Sequential understanding** using an LSTM network
* **Keyword‑based understanding** using a Bag‑of‑Words (BoW) vector

These two inputs are processed in parallel and merged to predict the most relevant medical intent, allowing the chatbot to respond with accurate, predefined first‑aid instructions.

---

## 🧠 Model Architecture

The chatbot uses a **dual‑input neural network** built with TensorFlow/Keras:

* **Input 1 (Text Sequences)**

  * Tokenization & padding
  * Embedding layer
  * LSTM (captures sentence context)

* **Input 2 (Bag‑of‑Words)**

  * Multi‑hot encoded keyword vector
  * Dense layer (captures critical medical terms)

* **Fusion & Output**

  * Concatenation of both branches
  * Fully connected layers
  * Softmax output for intent classification

### 📊 Architecture Diagram

![Dual Input Model Architecture](src/architecture.png)

---

## 📈 Training Performance

The model was trained for **50 epochs** using categorical cross‑entropy loss and the Adam optimizer.

* **Training Accuracy**: ~98%
* **Validation Accuracy**: ~70%
* **Trainable Parameters**: ~1 million

The training loss and accuracy trends are shown below:

![Training history Graph](src/training_history.png)

---

## 📁 Repository Structure

```bash
├── chatbot.py              # Chatbot inference & interaction logic
├── intents.json            # Dataset with intents, patterns, and responses
├── train_model.ipynb       # Model training and experimentation notebook
├── src/
│   ├── architecture.png    # Model architecture diagram
│   └── training_loss.png   # Training loss/accuracy graph
└── README.md               # Project documentation
```

---

## 📄 File Descriptions

### `intents.json`

Contains the **knowledge base** for the chatbot:

* `tag` – intent label (e.g., fever, cuts, snake_bite)
* `patterns` – example user queries
* `responses` – chatbot replies / first‑aid steps

---

### `train_model.ipynb`

Responsible for:

* Data preprocessing (tokenization, lemmatization)
* Feature engineering (Sequences + BoW)
* Dual‑input model construction
* Training, validation, and visualization
* Saving trained model & tokenizer

---

### `chatbot.py`

Implements:

* Loading trained model & preprocessing objects
* Real‑time user input handling
* Intent prediction with confidence threshold
* Response selection from `intents.json`

---

## ⚙️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NLTK**
* **NumPy**
* **Scikit‑learn**
* **Streamlit** (for UI, if deployed)

---

## 🚀 How to Run the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/medical-ai-chatbot.git
   cd medical-ai-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional)**

   ```bash
   jupyter notebook train_model.ipynb
   ```

4. **Run the chatbot**

   ```bash
   python chatbot.py
   ```

---

## 🔒 Safety & Ethics

* Uses a **confidence threshold** to avoid unsafe predictions
* Provides fallback responses when confidence is low
* Clearly informs users that it is **not a medical professional**

---

## 🔮 Future Improvements

* Expand and augment the dataset
* Reduce overfitting with better regularization
* Integrate transformer‑based models (e.g., BioBERT)
* Add multilingual support
* Deploy with a full web or mobile interface

---

⭐ *If you find this project useful, feel free to star the repository!*
