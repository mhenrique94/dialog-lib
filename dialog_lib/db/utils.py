import uuid
from .session import get_session
from .models import Chat


def create_chat_session(identifier=None, dbsession=get_session, model=Chat):
    if identifier is None:
        identifier = uuid.uuid4().hex

    with dbsession() as session:
        chat = session.query(model).filter_by(session_id=identifier).first()
        if not chat:
            chat = model(session_id=identifier)
            session.add(chat)

    return {"chat_id": chat.session_id}
