# add path to check for custom libraries
import sys
sys.path.append("/DialoGPT_test")

import torch
from fastapi import FastAPI
import uvicorn
from InstructionMessages import Instruction # local file so far

from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPTAssistant:

    def __init__(self):
        self.api = FastAPI(title="BVI Visual Assistant Generator API")

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.chat_history_ids = []
        self.history_counter = 0 

    def setup_routes(self):
        @self.api.post("/predict")
        async def predict(data:Instruction):
            #data = data.dict()
            instruction = data.instruction
            #new_user_input_ids = self.tokenizer.encode(instruction + self.tokenizer.eos_token, return_tensors='pt')

            new_user_input_ids = self.tokenizer.encode(instruction + self.tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if False else new_user_input_ids

            self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id, temperature=0.8)

            generated_text = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            self.history_counter += 1
            return {"answer": generated_text}

def main():
    visual_assistant = DialoGPTAssistant()
    visual_assistant.setup_routes()
    uvicorn.run(visual_assistant.api, host="0.0.0.0", port=80)



if __name__ == "__main__":
    main()