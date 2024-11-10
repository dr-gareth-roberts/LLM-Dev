import streamlit as st
import asyncio
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from llm_environment import LLMDevEnvironment
from llm_testing import ToolManager

st.set_page_config(page_title="LLM Development Environment", layout="wide")

async def init_environment():
    env = LLMDevEnvironment()
    await env.initialize_clients()
    return env

def main():
    st.title("LLM Development Environment")
    
    # Initialize session state
    if 'env' not in st.session_state:
        st.session_state.env = asyncio.run(init_environment())
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        selected_tools = st.multiselect(
            "Select Tools",
            ["openai", "anthropic", "langchain", "llama_index"],
            default=["openai", "anthropic"]
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
    
    # Main content
    st.header("Prompt Testing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area("Enter your prompt:", height=200)
        
        if st.button("Run Comparison"):
            if prompt:
                with st.spinner("Processing..."):
                    tool_manager = ToolManager(st.session_state.env)
                    results = asyncio.run(
                        tool_manager.run_comparison(prompt, selected_tools)
                    )
                    
                    for tool, response in results.items():
                        st.subheader(f"{tool} Response")
                        st.write(response)
            else:
                st.warning("Please enter a prompt.")
    
    with col2:
        st.subheader("Saved Prompts")
        prompts_dir = project_root / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        saved_prompts = list(prompts_dir.glob("*.txt"))
        if saved_prompts:
            selected_prompt = st.selectbox(
                "Load saved prompt:",
                [p.stem for p in saved_prompts]
            )
            
            if st.button("Load"):
                with open(prompts_dir / f"{selected_prompt}.txt") as f:
                    st.session_state.prompt = f.read()
                st.experimental_rerun()
        
        if prompt:
            prompt_name = st.text_input("Save prompt as:")
            if st.button("Save") and prompt_name:
                with open(prompts_dir / f"{prompt_name}.txt", "w") as f:
                    f.write(prompt)
                st.success("Prompt saved!")

if __name__ == "__main__":
    main()