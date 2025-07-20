from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

source = "data/RAG.pdf"
converter = DocumentConverter()

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])

docs = splitter.split_text(converter.convert(source).document.export_to_markdown())
print(docs)