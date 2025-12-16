from langchain_core.documents import Document
from datamodel.CoolChunk import CoolChunk


def generate_cool_chunks(chunks: list[Document]) -> list[CoolChunk]:
    """This function take a list of chunks (with the langcahin def) and transform every
       every chunk in a CoolChunk_

    Returns:
        list[CooLChunk]: the list of CoolChunks
    """
    if chunks is None:
        raise ValueError("There is no chunks provided please try again : \n")

    # Origin of the chunk
    file_name = chunks[0].metadata["source"]

    # List of cool chunks objects
    cool_chunks_list = []

    for chunk in chunks:

        docling_meta = chunk.metadata["dl_meta"]

        if docling_meta["doc_items"]:

            l_bbox = []
            l_npages = []

            for item in docling_meta["doc_items"]:

                if item["prov"]:

                    l_bbox.append(item["prov"][0]["bbox"])
                    l_npages.append(item["prov"][0]["page_no"])

        cool_chunks_list.append(
            CoolChunk(
                contextualized_chunk=chunk.page_content,
                bboxes=l_bbox,
                n_pages=l_npages,
                origin=file_name,
            )
        )
    return cool_chunks_list
