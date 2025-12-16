from utils.metadata_converter import meta_converter
import fitz
from datamodel.CoolChunk import CoolChunk

import os
import fitz
from utils.metadata_converter import meta_converter
from datamodel.CoolChunk import CoolChunk


def draw(chunks_with_metadata: list):
    list_cool_chunks: list[CoolChunk] = meta_converter(
        chunks_with_metadata["metadata"][0]
    )

    # Créer le dossier "highlighted_docs" dans le repo s'il n'existe pas
    output_dir = "highlighted_docs"
    os.makedirs(output_dir, exist_ok=True)

    # dictionnaire pour garder les docs ouverts
    docs_cache = {}

    for chunk in list_cool_chunks:
        origin = chunk.origin
        if origin not in docs_cache:
            docs_cache[origin] = fitz.open(origin)
        doc = docs_cache[origin]

        for page_no, bbox in zip(chunk.n_pages, chunk.bboxes):
            page_no0 = page_no - 1
            page = doc[page_no0]

            x1, x2 = bbox["l"], bbox["r"]
            y1, y2 = page.rect.height - bbox["t"], page.rect.height - bbox["b"]

            rect = fitz.Rect(x1, y1, x2, y2)
            page.add_highlight_annot(rect)

    # Sauvegarde tous les docs dans le dossier créé
    for origin, doc in docs_cache.items():
        output_file = os.path.join(
            output_dir, f"highlighted_{os.path.basename(origin)}"
        )
        doc.save(output_file)
        doc.close()

    print(f"PDFs sauvegardés dans le dossier '{output_dir}'")
