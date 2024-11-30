import assemblyai as aai
import json
import streamlit as st

from langchain.chains.llm  import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint

aai.settings.api_key = "api_key"


st.title("Audio Detection and Caption")


audio_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])

transcription_output = []

if audio_file:
    st.audio(audio_file, format="audio/mp3")  

    
    config = aai.TranscriptionConfig(speaker_labels=True)



    try:
       
        transcription_request = aai.Transcriber().transcribe(audio_file, config)

       
        while transcription_request.status != 'completed':
            st.write(f"Transcription status: {transcription_request.status}")
            transcription_request = aai.Transcriber().get_transcription(transcription_request.id)

        
        transcription_output = []

        for utterance in transcription_request.utterances:
            transcription_output.append({
                "speaker": utterance.speaker,
                "text": utterance.text
            })

        
        st.subheader("Transcription Output")
        st.json(transcription_output)

        st.success("Transcription complete!.")

    except Exception as e:
        st.error(f"An error occurred: {e}")








model_name = "facebook/bart-large-cnn"
llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token="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")



st.title("Summary of conversation")


text =transcription_output



prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\n\n{context}")]
)


llm_chain = LLMChain(llm=llm, prompt=prompt)

if text:
  
    summary = llm_chain.invoke({"context": text})
    
    
    summary_text = summary.get('text', summary) if isinstance(summary, dict) else summary

   
    if not isinstance(summary_text, str):
        summary_text = str(summary_text)

   

    st.write(summary_text)
