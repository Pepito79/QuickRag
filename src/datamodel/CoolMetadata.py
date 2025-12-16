from pydantic import BaseModel


class CoolMetadata(BaseModel):

    bboxes: list[dict]
    n_pages: list[int]
    origin: str
