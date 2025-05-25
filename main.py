from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect
from googletrans import Translator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from preloader import load_and_index_text

# Initialize FastAPI app
app = FastAPI()
translator = Translator()
@app.get("/")
async def root():
    return {"message": "API is live and working!"}
# Allow frontend and browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and embed your textbook
db = load_and_index_text()
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")

    # Detect language
    lang = detect(question)
    translated_question = question

    # Translate to English if Arabic
    if lang == "ar":
        translated_question = translator.translate(question, src="ar", dest="en").text

    # Get answer from QA model
    answer = qa_chain.run(translated_question)

    # Translate back to Arabic if needed
    if lang == "ar":
        answer = translator.translate(answer, src="en", dest="ar").text

    return {"answer": answer}
