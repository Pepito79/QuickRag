from langchain_docling import DoclingLoader
from typing import Iterable
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import smolvlm_picture_description
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import PictureDescriptionData
from typing_extensions import override
from utils.generate_cool_chunks import generate_cool_chunks


# Here we define a custom picture serialization strategy which leverages picture annotations
class AnnotationPictureSerializer(MarkdownPictureSerializer):
    @override
    def serialize(self, *, item, doc_serializer, doc, **kwargs):

        texts: list[str] = []
        for annotation in item:
            if isinstance(annotation, PictureDescriptionData):
                texts.append(f"Picture description: {annotation.text}")

        text_res = "\n".join(texts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


# Here we define a serializer provider that contains the table and Picture special serialization strategies
class MySerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=AnnotationPictureSerializer(),
            table_serializer=MarkdownTableSerializer(),
        )


class DoclingProcessor:

    def __init__(
        self,
        paths: str | Iterable[str],
        useOCR: bool = False,
        useSmolVLM: bool = False,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):

        # List of path
        self._paths = (
            paths
            if isinstance(paths, Iterable) and not isinstance(paths, str)
            else [paths]
        )

        # Set the tokenizer model that will be used to chunk the doc
        self._tokenizer = tokenizer_model

        # Set the parameters used
        self._useOCR = useOCR
        self._useSmolVLM = useSmolVLM
        self._pipeline = PdfPipelineOptions(
            do_ocr=self._useOCR,
            do_picture_description=self._useSmolVLM,
            do_table_structure=True,
        )

        # The prompt to get the image description
        if self._useSmolVLM:
            self._pipeline.picture_description_options = smolvlm_picture_description
            self._pipeline.picture_description_options.prompt = (
                "Describe the image in three sentences. Be consise and accurate."
            )

        # Create the document converter that we will use
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self._pipeline)
            }
        )

        # Define our custom Chunker
        self._chunker = HybridChunker(
            tokenizer=self._tokenizer, serializer_provider=MySerializerProvider()
        )

    def process(self):
        """Process the documents and create a list of CoolChunks to have access to some metadata
        Returns:
            list[CoolChunks ]: list of coolchunks obtained
        """
        # Create a list of cool chunks that contains text and metada
        try:
            loader = DoclingLoader(
                file_path=self._paths,
                converter=self._converter,
            )

            docs = loader.load()
            print("====== GENERATING YOUR COOL CHUNKS =======\n")
            print("************Processing shut*****************\n")
            cool_chunks = generate_cool_chunks(chunks=docs)
            print(
                f"==== Work finished : {len(cool_chunks)} CoolChunk generated ======="
            )
            return cool_chunks
        except Exception as e:
            raise Exception(f"There is an error while trying to load your docs:\{e}")
