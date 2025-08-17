# 🚀 Conversational RAG Chatbot  

An intelligent chatbot built with **LangChain**, **FAISS Vectorstore**, and **Streamlit**, deployed on **Render**.  
This chatbot allows users to upload documents (PDF, text, or URLs) and ask questions about them.  
If answers are not found in the documents, the bot fetches answers via **Google Search**.  

---

## ✨ Features  
✅ Upload multiple documents (PDF, DOCX, TXT, URLs)  
✅ Ask questions and get document-based answers  
✅ Falls back to **Google Search** when not found  
✅ Built using **LangChain + FAISS + HuggingFace models**  
✅ Deployed on **Render**  

---

## 🛠️ Tech Stack  

| **Component** | **Technology** |
|---------------|----------------|
| Frontend      | Streamlit 🎨   |
| Backend       | Flask / LangChain ⚙️ |
| Database      | FAISS Vectorstore 📂 |
| Deployment    | Render 🚀      |

---

## ⚡ Installation  

```bash
# 1️⃣ Clone the repo
git clone https://github.com/Riruru612/multi-source-rag-chatbot.git

# 2️⃣ Navigate into the project
cd <your-repo>

# 3️⃣ Create a virtual environment & activate it
python -m venv venv
source venv/bin/activate  # (Mac/Linux)
venv\Scripts\activate     # (Windows)

# 4️⃣ Install dependencies
pip install -r requirements.txt

# 5️⃣ Run the app
streamlit run app.py

📂 Project Structure

📦 project-root
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
├── /data                 # Uploaded documents
└── /vectorstore          # FAISS database
