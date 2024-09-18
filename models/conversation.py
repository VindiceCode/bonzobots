import pydantic

class ConversationPayload(pydantic.BaseModel):
    user_id: str
    message: str
    timestamp: float
