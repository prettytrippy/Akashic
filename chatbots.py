from chat import AkashicChatbot

def zephyr_chatbot():
  return AkashicChatbot("openbuddy-zephyr-7b-v14.1.Q4_K_M.gguf", format="zephyr", stop="User:")

def airoboros_chatbot():
  return AkashicChatbot("airoboros-l2-7b-3.0.Q4_K_M.gguf", format="airoboros", stop="[INST]")

def neural_chat_chatbot():
   return AkashicChatbot("neural-chat-7b-v3-2.Q5_K_M.gguf", format="intel", stop="### User:")
