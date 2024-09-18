import pytest
from models.conversation import ConversationPayload

def test_conversation_payload():
    payload = ConversationPayload(user_id='123', message='Hello', timestamp=1625097600.0)
    assert payload.user_id == '123'
    assert payload.message == 'Hello'
    assert payload.timestamp == 1625097600.0
