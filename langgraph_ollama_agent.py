import os
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
import tempfile
import base64
from PIL import Image as PILImage
import io

from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str


#model
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)

# Define node functions
def classification_node(state: State):
    '''จำแนกข้อความเป็นหมวดหมู่: ข่าว, บล็อก, งานวิจัย, หรืออื่นๆ'''

    prompt = PromptTemplate(
        input_variables=["text"],
        template="จำแนกข้อความต่อไปนี้เป็นหมวดหมู่: ข่าว, บล็อก, งานวิจัย, หรืออื่นๆ\n\nข้อความ:{text}\n\nหมวดหมู่:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

def entity_extraction_node(state: State):
    '''แยกชื่อเฉพาะ (คน, องค์กร, สถานที่) จากข้อความ'''

    prompt = PromptTemplate(
        input_variables=["text"],
        template="แยกชื่อเฉพาะ (คน, องค์กร, สถานที่) จากข้อความต่อไปนี้ แสดงผลลัพธ์เป็นรายการที่คั่นด้วยเครื่องหมายจุลภาค\n\nข้อความ:{text}\n\nชื่อเฉพาะ:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

def summarization_node(state: State):
    '''สรุปข้อความสำหรับหนึ่งประโยคสั้นๆ'''

    prompt = PromptTemplate(
        input_variables=["text"],
        template="สรุปข้อความต่อไปนี้ในหนึ่งประโยคสั้นๆ\n\nข้อความ:{text}\n\nสรุป:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

#Create tools and Build the workflow
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()


# Function to create a text-based workflow representation
def get_workflow_description():
    workflow_description = """
    ```mermaid
    graph TD
        A[Start] --> B[Classification]
        B --> C[Entity Extraction]
        C --> D[Summarization]
        D --> E[End]
    ```
    """
    return workflow_description

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Ollama Text Analysis Agent",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Ollama Text Analysis Agent")
    st.write("This agent analyzes text using a workflow of classification, entity extraction, and summarization.")
    
    # Sidebar for model settings
    with st.sidebar:
        st.header("Model Settings")
        model_name = st.selectbox(
            "Select Ollama Model",
            ["qwen2.5:7b", "llama3:8b", "codegemma:7b", "llava:7b"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.8, 0.1)
        
        st.header("Workflow Visualization")
        workflow_desc = get_workflow_description()
        st.markdown(workflow_desc)
    
    # Sample text selection
    st.header("Input Text")
    
    sample_option = st.radio(
        "Choose input method:",
        ["Use sample text", "Enter your own text", "Upload text file"]
    )
    
    if sample_option == "Use sample text":
        sample_text = """
        นางมนพร เจริญศรี รัฐมนตรีช่วยว่าการกระทรวงคมนาคม เปิดเผยว่า
        รัฐบาลและกระทรวงคมนาคม ได้พยายามผลักดันให้การประกาศใช้มาตรการค่าโดยสารรถไฟฟ้า 20 บาทตลอดสายทุกสาย 
        เริ่มตั้งแต่วันที่ 1 ตุลาคม 2568 ตามที่เคยประกาศไว้ แต่กระบวนการพิจารณากฎหมายที่เกี่ยวข้องทั้ง 3 ฉบับ 
        ได้แก่ พ.ร.บ. การรถไฟฟ้าขนส่งมวลชนแห่งประเทศไทย , พ.ร.บ.การบริหารจัดการระบบตั๋วร่วม 
        และ พ.ร.บ.การขนส่งทางราง รวมถึงการประกาศบังคับใช้ของกฎหมายลูก อาจจะเกิดความล่าช้า 
        คาดว่าจะใช้ระยะเวลาอีกประมาณ 1 เดือน จึงอาจทำให้ต้องขยับการประกาศใช้มาตรการออกไป
        """
        st.text_area("Sample Text", sample_text, height=200, key="input_text", disabled=True)
        input_text = sample_text
    
    elif sample_option == "Enter your own text":
        input_text = st.text_area("Enter your text here:", height=200, key="custom_text")
    
    else:  # Upload text file
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode("utf-8")
            st.text_area("File content:", input_text, height=200, disabled=True)
        else:
            input_text = ""
    
    # Process button
    if st.button("Analyze Text") and input_text:
        with st.spinner("Processing text..."):
            # Update model settings if changed
            global llm
            llm = ChatOllama(model=model_name, temperature=temperature)
            
            # Process the text
            state_input = {"text": input_text}
            try:
                result = app.invoke(state_input)
                
                # Display results in tabs
                st.header("Analysis Results")
                tab1, tab2, tab3 = st.tabs(["Classification", "Entity Extraction", "Summary"])
                
                with tab1:
                    st.subheader("Text Classification")
                    st.info(result["classification"])
                
                with tab2:
                    st.subheader("Entities Extracted")
                    if result["entities"]:
                        for entity in result["entities"]:
                            st.write(f"• {entity}")
                    else:
                        st.write("No entities found.")
                
                with tab3:
                    st.subheader("Text Summary")
                    st.success(result["summary"])
                
                # Save results option
                st.download_button(
                    label="Download Results",
                    data=f"""Text Analysis Results
                    
Classification: {result["classification"]}

Entities: {', '.join(result["entities"])}

Summary: {result["summary"]}
                    """,
                    file_name="text_analysis_results.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
    elif not input_text and st.button("Analyze Text"):
        st.warning("Please enter or upload some text to analyze.")

if __name__ == "__main__":
    main()