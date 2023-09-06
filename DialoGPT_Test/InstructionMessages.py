from pydantic import BaseModel

class Instruction(BaseModel):
    instruction: str
    model_id: int