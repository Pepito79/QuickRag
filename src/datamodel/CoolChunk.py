from pydantic import BaseModel


class CoolChunk(BaseModel):

    contextualized_chunk: str
    bboxes: list[dict]
    n_pages: list[int]
    origin: str
