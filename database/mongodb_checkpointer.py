from contextlib import contextmanager
from collections.abc import Generator
from langgraph.checkpoint.mongodb import MongoDBSaver
from database.mongodb import MONGO_URI

@contextmanager
def get_checkpointer() -> Generator[MongoDBSaver, None, None]:
    with MongoDBSaver.from_conn_string(
        MONGO_URI,
        db_name="your_db",
    ) as checkpointer:
        yield checkpointer