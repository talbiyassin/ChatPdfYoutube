import streamlit as st
from dotenv import load_dotenv
import pickle
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()


def get_youtube_transcript(video_url):
    try:
        # Extract video ID from the URL
        video_id = video_url.split('v=')[-1].split('&')[0]

        # Retrieve the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Extract text from the transcript
        transcript_text = ' '.join([entry['text'] for entry in transcript])

        return transcript_text

    except Exception as e:
        return f"Error: {str(e)}"


def main():
    st.title("PDF 📑 & YouTube 🎞️ Chatbot 💬")
    st.write("🚀Projet d'Analyse Vidéo et PDF 🧐💬")
    st.write("Explorons le contenu de manière interactive")

    # Use st.sidebar for left alignment
    with st.sidebar:
        # Add your logo with reduced size
        st.image("logo.png", width=250, use_column_width=False)

        # Main content area
        st.title("Bienvenue dans notre Projet!")
        st.write(
            "Notre projet vise à faciliter une compréhension approfondie en permettant aux utilisateurs de poser des questions "
            "et d'obtenir des réponses à partir de vidéos YouTube ou de documents PDF. 💡"
        )

        # Choose PDF or YouTube
        option = st.radio("⚠️Sélectionnez une option:⚠️", ["PDF", "YouTube"])

    if option == "PDF":
        # Upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type='pdf')

        if pdf is not None:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")

            if query:
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                st.subheader("Response:")
                st.write(response)

    elif option == "YouTube":
        # Take YouTube video URL as input
        video_url = st.text_input("Enter the YouTube video URL:")

        # Get transcript and display result
        transcript_result = get_youtube_transcript(video_url)

        if not transcript_result.startswith("Error"):
            st.subheader("Transcript:")
            #st.write(transcript_result)
            st.subheader("Chat:")
            # Process the transcript text similar to the PDF
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            transcript_chunks = text_splitter.split_text(text=transcript_result)
            video_store_name = video_url.split('v=')[-1].split('&')[0]

            if os.path.exists(f"{video_store_name}.pkl"):
                with open(f"{video_store_name}.pkl", "rb") as f:
                    VideoVectorStore = pickle.load(f)
            else:
                # Initialize 'embeddings' variable here
                embeddings = OpenAIEmbeddings()
                VideoVectorStore = FAISS.from_texts(transcript_chunks, embedding=embeddings)
                with open(f"{video_store_name}.pkl", "wb") as f:
                    pickle.dump(VideoVectorStore, f)

            # Continue with the same question answering logic as in the PDF section
            query_video = st.text_input("Ask questions about your video:")

            if query_video:
                video_docs = VideoVectorStore.similarity_search(query=query_video, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response_video = chain.run(input_documents=video_docs, question=query_video)
                st.subheader("Response from Video:")
                st.write(response_video)

        else:
            st.subheader("Error:")
            st.write(transcript_result)

    # Custom footer
    footer = """
        <footer style="margin-top: 100px;">
            <div>
                <p style="font-size: 1.1rem;">
                   🏫 Made by Sesame Student 👨‍🎓👨‍🎓👩‍🎓
                </p>
            </div>
        </footer>
    """
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == '__main__':
    main()


