from datamodel.CoolMetadata import CoolMetadata
import json


def _meta_converter(meta_text: dict) -> CoolMetadata:

    return CoolMetadata(
        bboxes=json.loads(meta_text["bboxes"]),
        n_pages=json.loads(meta_text["n_pages"]),
        origin=meta_text["file_origin"],
    )


def meta_converter(metas: list[dict]):

    if metas is None:
        raise Exception("Exception raised : there are no metadatas")

    list_coolMetadata = []
    for meta in metas:

        list_coolMetadata.append(_meta_converter(meta))

    return list_coolMetadata
