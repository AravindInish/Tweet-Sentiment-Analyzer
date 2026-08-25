# 🐦 Tweet Sentiment Analyzer

### 🧠 Deep Learning • NLP • Sentiment Analysis • Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-8A2BE2?style=for-the-badge" />
</p>

<p align="center">
  <b>Turn raw tweets into meaningful sentiment predictions using Deep Learning.</b>
</p>

---

## 🚀 Overview

**Tweet Sentiment Analyzer** is an NLP-based Deep Learning project that classifies tweets into **Positive** or **Negative** sentiment.

The model is trained using the **Sentiment140 dataset containing 1.6 million tweets**, followed by text cleaning, tokenization, sequence padding, neural-network training, and evaluation.

The trained model and tokenizer can then be used inside a **Streamlit application**, allowing users to enter a tweet and receive a sentiment prediction in real time.

> 💡 **From Tweet → Text Processing → Neural Network → Sentiment Prediction**

---

## ✨ Key Features

| Feature                      | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| 📊 **Large Dataset**         | Trained on 1.6M tweets                               |
| 🧹 **Text Cleaning**         | Removes URLs, mentions, special characters and noise |
| 🔤 **Tokenization**          | Converts text into numerical sequences               |
| 📏 **Sequence Padding**      | Standardizes tweet length to 100 tokens              |
| 🧠 **Deep Learning**         | TensorFlow/Keras neural-network architecture         |
| ⚡ **Fast Prediction**        | Saved model can be loaded without retraining         |
| 🌐 **Streamlit UI**          | Interactive web interface                            |
| 🎯 **Binary Classification** | Positive vs Negative sentiment                       |
| 📈 **Training Monitoring**   | Accuracy and loss tracking                           |
| 💾 **Model Export**          | Saves trained model and tokenizer                    |

---

# 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │   Sentiment140       │
                  │    1.6M Tweets       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Data Preprocessing │
                  │                      │
                  │ • Remove URLs        │
                  │ • Remove @mentions   │
                  │ • Clean hashtags     │
                  │ • Remove symbols     │
                  │ • Lowercase text     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Tokenizer       │
                  │   Vocabulary = 50K   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Sequence Padding   │
                  │    Length = 100      │
                  └──────────┬───────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │       Deep Learning Model    │
             │                               │
             │       Embedding (128)        │
             │              ↓                │
             │   GlobalAveragePooling1D      │
             │              ↓                │
             │        Dense (64, ReLU)       │
             │              ↓                │
             │          Dropout (0.5)        │
             │              ↓                │
             │       Dense (1, Sigmoid)      │
             └───────────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Sentiment Prediction │
                  │                      │
                  │ 😔 Negative          │
                  │ 😐 Mixed / Neutral   │
                  │ 🎉 Positive          │
                  └──────────────────────┘
```

---

# 📊 Dataset

The project uses the **Sentiment140 dataset**, downloaded through KaggleHub.

### Dataset Statistics

* 📝 **Total Tweets:** 1,600,000
* 😔 **Negative:** 800,000
* 😊 **Positive:** 800,000
* 🧾 **Features:** `target`, `id`, `date`, `flag`, `user`, `text`
* ⚖️ **Class Distribution:** Balanced

The notebook confirms 1.6M records with no missing values and an even 800K/800K target distribution.

---

# 🧹 Data Preprocessing

Raw tweets contain significant noise, so the project applies a cleaning pipeline before training.

### Cleaning Pipeline

```text
Raw Tweet
    │
    ├── Remove URLs
    │
    ├── Remove @mentions
    │
    ├── Remove hashtag symbols
    │
    ├── Remove special characters & numbers
    │
    ├── Remove extra whitespace
    │
    └── Convert to lowercase
            │
            ▼
       Cleaned Tweet
```

The original labels are transformed as:

```text
0 → Negative 😔
4 → Positive 😊
```

This produces a binary sentiment target of `0` and `1`.

---

# 🔤 NLP Pipeline

The cleaned text is converted into numerical representations using a Keras tokenizer.

### Configuration

```python
VOCAB_SIZE = 50_000
MAX_SEQUENCE_LENGTH = 100
```

The notebook converts tweets into integer sequences and pads/truncates them to a fixed length of 100 tokens.

### Dataset Split

```text
1,600,000 Tweets
        │
        ├─────────────── 80% ───────────────┐
        │                                   │
        ▼                                   ▼
  1,280,000 Training                   320,000 Testing
```

The split uses stratification with `random_state=42`.

---

# 🧠 Deep Learning Model

The project uses a lightweight neural architecture built with **TensorFlow/Keras**.

### Architecture

```text
Input
  │
  ▼
Embedding
50,000 vocabulary × 128 dimensions
  │
  ▼
GlobalAveragePooling1D
  │
  ▼
Dense
64 neurons + ReLU
  │
  ▼
Dropout
0.5
  │
  ▼
Dense
1 neuron + Sigmoid
  │
  ▼
Sentiment Score
```

The model is compiled using:

```python
optimizer = Adam(learning_rate=0.001)
loss = "binary_crossentropy"
metric = "accuracy"
```

---

# ⚙️ Training

The notebook trains the model using:

```python
BATCH_SIZE = 1024
EPOCHS = 1
```

Training was performed with a **10% validation split** from the training data.

### Recorded Training Result

```text
Training Accuracy     ≈ 76.32%
Validation Accuracy   ≈ 76.78%

Training Loss         ≈ 0.5031
Validation Loss       ≈ 0.4814
```

The recorded training run completed in approximately 114 seconds.

> ⚠️ The notebook contains the evaluation code, but the test-evaluation output itself is not recorded in the supplied notebook. Therefore, no unsupported test-accuracy number is claimed here.

---

# 💾 Model Export

After training, the project saves both the neural network and tokenizer:

```text
sentiment_model.h5
tokenizer.pickle
```

These files allow the application to perform predictions without retraining the model.

---

# 🌐 Streamlit Application

The project includes a **Streamlit-based Tweet Sentiment Analyzer**.

### User Flow

```text
             👤 User
                │
                ▼
       ✍️ Enter Tweet
                │
                ▼
         🧹 Clean Text
                │
                ▼
          🔤 Tokenize
                │
                ▼
          📏 Pad Sequence
                │
                ▼
       🧠 Neural Network
                │
                ▼
       🎯 Sentiment Score
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
    😔 Neg    😐 Mixed   🎉 Pos
```

The application loads the saved model and tokenizer, preprocesses user input, performs inference, and displays the resulting sentiment score.

### Sentiment Thresholds

```text
Score ≤ 0.40      → 😔 Negative
0.40–0.60         → 😐 Neutral / Mixed
Score ≥ 0.60      → 🎉 Positive
```

---

# 🛠️ Tech Stack

<p align="center">

| Technology        | Purpose                             |
| ----------------- | ----------------------------------- |
| 🐍 **Python**     | Core programming language           |
| 🐼 **Pandas**     | Data manipulation                   |
| 🔢 **NumPy**      | Numerical operations                |
| 🧠 **TensorFlow** | Deep Learning                       |
| 🔥 **Keras**      | Neural-network API                  |
| 📊 **Matplotlib** | Training visualization              |
| 🌐 **Streamlit**  | Web application                     |
| 📦 **KaggleHub**  | Dataset download                    |
| 🚇 **Pyngrok**    | Public tunneling during development |

</p>

---

# 📁 Project Structure

```text
📦 Tweet-Sentiment-Analyzer
│
├── 📓 Sentiment140_.ipynb
│
├── 🐍 app.py
│
├── 🧠 sentiment_model.h5
│
├── 🔤 tokenizer.pickle
│
└── 📄 README.md
```

---

# 🚀 Run Locally

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
cd <YOUR-REPOSITORY>
```

## 2️⃣ Install Dependencies

```bash
pip install tensorflow pandas numpy matplotlib streamlit
```

## 3️⃣ Make Sure These Files Exist

```text
app.py
sentiment_model.h5
tokenizer.pickle
```

## 4️⃣ Launch the Application

```bash
streamlit run app.py
```

The notebook explicitly uses `streamlit run app.py` to launch the application.

---

# 🧪 Example

### Input

```text
I absolutely love this product! It made my day ❤️
```

### Output

```text
🎉 Positive Sentiment!

Score: 0.92
```

Another example:

```text
This is the worst experience ever.
```

```text
😔 Negative Sentiment

Score: 0.18
```

---

# 📈 Project Highlights

<div align="center">

| 📌 Metric              |         Value |
| ---------------------- | ------------: |
| 📝 Tweets              | **1,600,000** |
| 😔 Negative Samples    |   **800,000** |
| 😊 Positive Samples    |   **800,000** |
| 🔤 Vocabulary Size     |    **50,000** |
| 📏 Sequence Length     |       **100** |
| 🧠 Embedding Dimension |       **128** |
| 🏋️ Batch Size         |     **1,024** |
| 🔄 Epochs              |         **1** |
| 📊 Validation Accuracy |    **76.78%** |

</div>

---

# 🔮 Future Improvements

This project provides a strong baseline, but there is plenty of room to push the system further.

### 🚀 Planned Enhancements

* [ ] 🔥 Train for multiple epochs
* [ ] 🧠 Experiment with LSTM / GRU
* [ ] 🤗 Add Transformer-based models
* [ ] 📊 Add confusion matrix
* [ ] 📈 Add precision, recall and F1-score
* [ ] 🔍 Add model explainability
* [ ] 🌍 Support multilingual sentiment
* [ ] ⚡ Optimize inference latency
* [ ] 📱 Improve Streamlit UI/UX
* [ ] ☁️ Deploy the application publicly
* [ ] 🧪 Add automated model evaluation
* [ ] 🗃️ Add experiment tracking

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

```text
Python
   ↓
Data Analysis
   ↓
Natural Language Processing
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Deep Learning
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

It is designed as an end-to-end NLP workflow rather than just a standalone model-training experiment.

---

# ⚠️ Limitations

The current implementation has some important limitations:

* The model is trained for only **one epoch**, so additional training may improve performance.
* The preprocessing pipeline removes numbers and many special characters, which can sometimes carry sentiment information.
* The model uses `GlobalAveragePooling1D`, which is simpler than sequence-aware architectures such as LSTM, GRU, or Transformers.
* The notebook does not contain a recorded test-set accuracy result.
* The Streamlit classifier is fundamentally trained around positive/negative labels, while the UI also exposes a mixed/neutral range based on prediction thresholds.

---

# 🤝 Contributing

Contributions are welcome! 🎉

```bash
# Fork → Clone → Create Branch → Make Changes → Pull Request
```

If you have ideas for improving the model, UI, preprocessing pipeline, or deployment architecture, feel free to contribute.

---

# 📜 License

This project is intended for **educational and research purposes**.

Please review the original dataset's licensing and usage conditions before redistributing the dataset or derived assets.

---

# 👨‍💻 Author

### **Aravind**

🚀 AI / ML Developer
🧠 Deep Learning & NLP Enthusiast
💻 Full-Stack & Product Development
📚 Continuous Learner

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Made with 🧠 + 💻 + ☕**

</div>
