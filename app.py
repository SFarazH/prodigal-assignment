
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import torch
import re

# --- regex patterns ---
profanity = ['fuck','hell','damn','f***','shit','bullshit','crap','ass','asshole','dumbass','stupid', 'shitload']
censored_patterns = [
    r"f[\W_]*[u\*]+[\W_]*[c\*]+[\W_]*[k\*]+",
    r"s[\W_]*[h\*]+[\W_]*[i\*]+[\W_]*[t\*]+",
    r"d[\W_]*[a\*]+[\W_]*[m\*]+[\W_]*[n\*]*",
]
sensitive_info_patterns = [
    r'\$\s*\d+',                         
    r'\d+\s*(dollars|usd)',              
    r'amount\s*(is|of)?\s*\$?\s*\d+',    
    r'balance\s*(is|of)?\s*\$?\s*\d+',   
    r'outstanding balance of \$?\d+',
    r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b.*\b(dollars|usd)\b'

]
identity_verification_patterns = [
    r'\bdate of birth\b',
    r'\bD\.?O\.?B\.?\b',
    r'\baddress\b',
    r'\bSSN\b',
    r'\bsocial security (number)?\b'
]

# --- Load LLM model ---
@st.cache_resource
def load_profanity_model():
    model = T5ForConditionalGeneration.from_pretrained("faraz02/profanity-checker")
    tokenizer = T5Tokenizer.from_pretrained("faraz02/profanity-checker")
    return model, tokenizer

profanity_model, profanity_tokenizer = load_profanity_model()
label_map_profanity = {'yes': 1, 'no': 0}

def load_compliance_model():
  model = T5ForConditionalGeneration.from_pretrained("faraz02/compliance-checker")
  tokenizer = T5Tokenizer.from_pretrained("faraz02/compliance-checker")
  return model, tokenizer

compliance_model, compliance_tokenizer = load_compliance_model()
label_map_compliance = {'Yes': 1, 'No': 0}

# --- Define Functions ---
def contains_profanity_regex(convo):
    profanity_speaker = []
    for message in convo:
        text = message['text'].lower()
        for word in profanity:
            pattern = rf"\b{re.escape(word)}(s|es|ed|ing)?\b"
            if re.search(pattern, text):
                profanity_speaker.append(message['speaker'])
        for pattern in censored_patterns:
            if re.search(pattern, text):
                profanity_speaker.append(message['speaker'])
    return list(set(profanity_speaker))

def contains_pattern(text, patterns):
    return any(re.search(pat, text, re.IGNORECASE) for pat in patterns)

def check_violation(dialogue):
    verification_done = False
    for turn in dialogue:
        if turn['speaker'].lower() == 'agent':
            if contains_pattern(turn['text'], identity_verification_patterns):
                verification_done = True
            elif contains_pattern(turn['text'], sensitive_info_patterns):
                if not verification_done:
                    return True
    return False

def preprocess_input_for_comp_llm(conversation):
    intro = (
        "Given the following conversation, determine whether the agent has shared sensitive information "
        "(like account balance in words or dollars or account details) without verifying the customer's identity (such as date of birth, "
        "address, or Social Security Number). Respond with \"True\" if a violation has occurred, otherwise \"False\".\n\nConversation:\n"
    )

    dialogue = ""
    for turn in conversation:
        dialogue += f"{turn['speaker']}: {turn['text']}\n"

    final_input = intro + dialogue
    return final_input

def profanity_llm(data):
    profanity_speakers = []
    predictions = []
    for entry in data:
        text = entry['text']
        speaker = entry['speaker']
        input_ids = profanity_tokenizer("classify: " + text, return_tensors="pt", padding=True, truncation=True).input_ids
        with torch.no_grad():
            output = profanity_model.generate(input_ids, max_length=2)
        pred = profanity_tokenizer.decode(output[0], skip_special_tokens=True).lower().strip()
        if label_map_profanity.get(pred, 0) == 1:
            profanity_speakers.append(speaker)
        predictions.append(label_map_profanity.get(pred, 0))
    return list(set(profanity_speakers))

def compliance_llm(data):
  inputs = compliance_tokenizer(data, return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
    outputs = compliance_model.generate(**inputs, max_length=5)
  predicted_text = compliance_tokenizer.decode(outputs[0], skip_special_tokens=True)
  return predicted_text



# --- app ---
st.title("Conversation Checker 🚀")

uploaded_file = st.file_uploader("Upload JSON File", type="json")

if uploaded_file is not None:
    try:
        uploaded_content = uploaded_file.read()
        data = json.loads(uploaded_content.decode('utf-8'))
        st.session_state['uploaded_data'] = data
        if not isinstance(data, list):
            st.error("JSON should be an array of objects!")
            st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON file!")
        st.stop()
    st.success("File uploaded successfully!")
    check_type = st.selectbox("What do you want to check?", ("Profanity", "Compliance"))
    method = st.selectbox("Which method do you want to use?", ("Regex", "LLM"))

    if st.button("Predict"):
        data = st.session_state['uploaded_data']
        if check_type == "Profanity":
            if method == "Regex":
                speakers = contains_profanity_regex(data)
                if speakers:
                    st.error(f"⚠️ Profanity detected by: {',  '.join(speakers)}")
                else:
                    st.success("✅ No profanity detected.")
            else:
                speakers = profanity_llm(data)
                if speakers:
                    st.error(f"⚠️ Profanity detected by: {',  '.join(speakers)}")
                else:
                    st.success("✅ No profanity detected.")

        elif check_type == "Compliance":
          if method=='Regex':
            violated = check_violation(data)
            if violated:
              st.error("⚠️ Compliance Violation Detected!")
            else:
              st.success("✅ No Compliance Violations Detected.")
          else:
            preprocess_data = preprocess_input_for_comp_llm(data)
            violated = compliance_llm(preprocess_data)
            if violated:
              st.error("⚠️ Compliance Violation Detected!")
            else:
              st.success("✅ No Compliance Violations Detected.")

