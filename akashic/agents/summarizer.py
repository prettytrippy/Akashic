from akashic.agents.chatbot import AkashicChatbot
from akashic.model import AkashicModel
from akashic.utils import read_file, count_tokens
import re
from tqdm import tqdm

default_summarizer_message = f"You are a chatbot assistant who summarizes documents."

class AkashicSummarizer(AkashicChatbot):
    def __init__(self, model, system_message=default_summarizer_message, divider=4):
        super().__init__(model, system_message=system_message)
        self.divider = divider
    
    def chunk_text(self, text):
        sections = re.split(r'\. |\n|\r|\t|\! |\? |\: |\;', text)
        sections = [section.strip() for section in sections if section.strip()]

        combined_strings = []
        current_string = ""

        for section in sections:
            if count_tokens(current_string) + count_tokens(section) < (self.get_context_length() // self.divider):
                current_string += section
            else:
                combined_strings.append(current_string)
                current_string = section

        combined_strings.append(current_string) 
        return combined_strings
    
    def make_prompt(self, user_input):
        return f"""Briefly summarize the following passage. 
        Focus on important information, as if you're taking notes for an important exam. 
        Just reply with the summary, no extra text like "Here's a summary of the passage".
        Here's the passage:\n\n{user_input}"""

    def summarize(self, text):
        self.messages = []
        results = []
        chunks = self.chunk_text(text)
        for chunk in (chunks):#, desc="Summarizing text..."):
            msg = self.make_prompt(chunk)
            result = self.send_prompt(msg, stream=False)
            result = list(result)[0]
            results.append(result)
            
        return "\n\n".join(results)
    
    def summarize_file(self, filepath):
        text = read_file(filepath)
        summary = self.summarize(text)
        return summary