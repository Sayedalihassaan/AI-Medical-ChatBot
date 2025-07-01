# 🤖 AI-Medical-ChatBot

An AI-powered Medical ChatBot designed to assist users with reliable, informative responses based on medical literature. This project uses advanced Natural Language Processing (NLP) techniques and integrates RAG (Retrieval-Augmented Generation) pipelines to provide contextually accurate answers.

---

## 📁 Project Structure

```

├── Datasets/
│   └── The-Gale-Encyclopedia-of-Medicine.csv
├── Notebook/
│   └── AI\_Medical\_ChatBot.ipynb
├── System/
│   ├── helper.py
│   ├── stcture.py
│   ├── store\_db.py
│   ├── temp.py
│   ├── frontend.py
│   ├── .env.example
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE

````

---

## 🚀 Features

- 💬 Chat interface with AI-driven medical responses
- 📚 RAG-based backend for retrieving and generating accurate answers
- 📖 Trained on **The Gale Encyclopedia of Medicine**
- 🔍 Semantic search capabilities using vector embeddings
- 🌐 Flask or Streamlit-based front-end (if applicable)
- 🔐 Environment variables for API key security

---

## 🧠 Technologies Used

- Python 3.10+
- LangChain
- Google Generative AI
- FAISS (vector store)
- Pandas, NumPy
- Streamlit or Flask (for frontend)
- dotenv

---

## ⚙️ Installation

```bash
git clone https://github.com/Sayedalihassaan/AI-Medical-ChatBot.git
cd AI-Medical-ChatBot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
````

---

## 🔑 Setup `.env`

Rename `.env.example` to `.env` and add your credentials:

```
GOOGLE_API_KEY=your_google_api_key
```

---

## 🧪 Usage

### 1. Run the Notebook (for demo and testing):

```bash
jupyter notebook Notebook/AI_Medical_ChatBot.ipynb
```

### 2. Run the Full System:

```bash
python System/frontend.py
```

This will launch the chatbot interface where you can start asking medical questions.

---

## 📝 Example Question

> **User:** What are the symptoms of diabetes?
> **Bot:** Based on The Gale Encyclopedia of Medicine, the common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision...

---

## ✅ To-Do

* [ ] Add user feedback mechanism
* [ ] Improve UI for better UX
* [ ] Integrate more medical sources
* [ ] Dockerize the application

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍⚕️ Disclaimer

This chatbot is **not** a replacement for professional medical advice. Always consult with a licensed healthcare provider for medical concerns.

---

## 🙋‍♂️ Author

**Sayed Ali Hassan**
[GitHub](https://github.com/Sayedalihassaan)

---

```

---

✅ **Next Step:** Copy the content above into your `README.md` file. Let me know if you also want a logo/banner or deployment instructions (e.g., on Streamlit, HuggingFace Spaces, or Docker).
```
