import re
import time
from datetime import date
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                         pipeline, GPT2Tokenizer, GPT2LMHeadModel)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st

st.set_page_config(page_title="Curiosity-16", layout="centered", initial_sidebar_state="collapsed")
st.title('Curiosity-16')
ts = int(time.time())
today = date.today()
print(today)

st.write("A friendly assistant. (Research Model -- 2025)")
st.caption("""You are visiting on:    """ + str(today))

# Loading Curiosity-16 + AutoTokenizer
model_id = "ariankharazmi/Curiosity-16"
@st.cache_resource(show_spinner=True)
def _load_text_generator():
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.to(device_str)
    model.eval()
    torch.set_grad_enabled(False)

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=3,
        length_penalty=1.8,
        min_new_tokens=8,
        max_new_tokens=132,
        no_repeat_ngram_size=2,
        repetition_penalty=1.15,
        return_full_text=False,
    )
text_generator = _load_text_generator()

# System Prompt for C16 -- Yes, this does impact response quality.
system_prompt = (
    "You are a concise factual assistant named Curiosity-16. "
    "Rules: "
    "• Answer in 1–2 plain sentences. No lists unless asked. "
    "• Do not write emails, salutations, or ask the user questions back. "
    "• If unsure, say “I’m not sure.” "
    "• If asked for illegal or harmful instructions, refuse briefly and suggest a safe alternative.\n\n")

user_prompt = st.text_input("", key="prompt", placeholder="Ask me anything!")
submit = st.button("Go")

def _two_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:2]).strip()
if submit and user_prompt.strip():
    with st.spinner("Processing..."):
        prompt = system_prompt + user_prompt.strip()
        out = text_generator(prompt)[0]["generated_text"].strip()
        st.markdown(_two_sentences(out))
st.write("---")
st.caption("Curiosity-16 can make mistakes. Verify important information.")
st.markdown("**Curiosity-16.2025.08.07 -- Built by Arian Kharazmi**")
with st.sidebar.expander("Documentation"):
    st.markdown("[EEP2025 (Research co-op) Deliverables](https://github.com/ariankharazmi/EEP25Deliverables)")
    st.markdown("[Curiosity-16 LLM HuggingFace Repo](https://huggingface.co/ariankharazmi/Curiosity-16/tree/main)")
    st.markdown("[Curiosity-16 LLM GitHub](https://github.com/ariankharazmi/Curiosity-16-LLM)")
    st.markdown("[Curiosity LLM (All Models)](https://github.com/ariankharazmi/curiosity-llm)")
#t1, t2 = st.columns(2)







