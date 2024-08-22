from chat import AkashicChatbot, default_system_prompt

# max context: 4096
def zephyr_chatbot(system_prompt=default_system_prompt, context_length=1024):
  return AkashicChatbot("openbuddy-zephyr-7b-v14.1.Q4_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="zephyr")

def airoboros_chatbot(system_prompt=default_system_prompt, context_length=1024):
  return AkashicChatbot("airoboros-l2-7b-3.0.Q4_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="airoboros")

# max context: 1024
def neural_chat_chatbot_Q5(system_prompt=default_system_prompt, context_length=2048):
   return AkashicChatbot("neural-chat-7b-v3-3.Q5_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="intel")

def neural_chat_chatbot_Q4(system_prompt=default_system_prompt, context_length=2048):
   return AkashicChatbot("neural-chat-7b-v3-3.Q4_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="intel")

# max context: 8192
def stable_lm_chatbot(system_prompt=default_system_prompt, context_length=1024):
  return AkashicChatbot("stablelm-zephyr-3b.Q5_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="stablelm")

# max context: 32768
def tiny_llama_chatbot_10(system_prompt=default_system_prompt, context_length=8192):
  return AkashicChatbot("tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="zephyr")

def metamath_chatbot(system_prompt=default_system_prompt, context_length=2048):
  return AkashicChatbot("metamath-cybertron-starling.Q4_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="chatml")

def llama3_Q4_chatbot(system_prompt=default_system_prompt, context_length=2048):
  return AkashicChatbot("Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="llama_3")

def textbase_Q4_chatbot(system_prompt=default_system_prompt, context_length=512):
  return AkashicChatbot("TextBase-7B-v0.1.Q4_K_M.gguf", system_prompt=system_prompt, context_length=context_length, format="chatml")


