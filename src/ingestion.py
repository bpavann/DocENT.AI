from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader


class IngestionAgent:
    def load_all_docs(self,data_dir: str) -> List[Any]:
        """
        Load all supported files from the data directory 
        Convert to LangChain document structure.
        Supported: PDF, TXT, CSV, Excel, Word.
        """
        # Root data folder
        data_path = Path(data_dir).resolve()
        print(f"Status: Data path: {data_path}")
        
        #defined empty document list
        documents = []

        # PDF files
        pdf_files = list(data_path.glob('**/*.pdf'))
        print(f"Status: Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
        for pdf_file in pdf_files:
            print(f"Status: Loading PDF: {pdf_file}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                loaded = loader.load()
                print(f"Status: Loaded {len(loaded)} PDF docs from {pdf_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

        # Word files
        docx_files = list(data_path.glob('**/*.docx'))
        print(f"Status: Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
        for docx_file in docx_files:
            print(f"Status: Loading Word: {docx_file}")
            try:
                loader = Docx2txtLoader(str(docx_file))
                loaded = loader.load()
                print(f"Status: Loaded {len(loaded)} Word docs from {docx_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load Word {docx_file}: {e}")

        # CSV files
        csv_files = list(data_path.glob('**/*.csv'))
        print(f"Status: Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
        for csv_file in csv_files:
            print(f"Status: Loading CSV: {csv_file}")
            try:
                loader = CSVLoader(str(csv_file))
                loaded = loader.load()
                print(f"Status: Loaded {len(loaded)} CSV docs from {csv_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

        # Excel files
        xlsx_files = list(data_path.glob('**/*.xlsx'))
        print(f"Status: Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
        for xlsx_file in xlsx_files:
            print(f"Status: Loading Excel: {xlsx_file}")
            try:
                loader = UnstructuredExcelLoader(str(xlsx_file))
                loaded = loader.load()
                print(f"Status: Loaded {len(loaded)} Excel docs from {xlsx_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

        # TXT files
        txt_files = list(data_path.glob('**/*.txt'))
        print(f"Status: Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
        for txt_file in txt_files:
            print(f"Status: Loading TXT: {txt_file}")
            try:
                loader = TextLoader(str(txt_file))
                loaded = loader.load()
                print(f"Status: Loaded {len(loaded)} TXT docs from {txt_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

        
        print(f"Status: Total loaded documents: {len(documents)}")
        return documents

#To check the instant output
if __name__ == "__main__":
    ingest=IngestionAgent()
    docs = ingest.load_all_docs("/Users/pavankumarb/Documents/My Learning/DocENTmcp/data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)