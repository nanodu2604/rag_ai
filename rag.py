from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
#from llama_parse import LlamaParse
from llama_index.core.embeddings import resolve_embed_model
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class ChatPDF:
    vector_index = None
    retriever = None
    vector_index = None

    def __init__(self):
        # set up the language model and text spliter
        self.llm = Ollama(model='mistral', request_timeout=30.0)
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        template = ("""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise.  
            Question: {query_str} \n
            Context: {context_str} \n
            Answer: 
            """)
        self.qa_template = PromptTemplate(template)

    def ingest(self, file_path: str):
        # Borrow the Llama Parse library process the file
        #parser = LlamaParse()
        #file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_files=[file_path]#, file_extractor=file_extractor
        ).load_data()

        # Resolve embedding model and create a VectorStoreIndex from documents
        embed_model = resolve_embed_model("local:BAAI/bge-m3")
        self.vector_index = VectorStoreIndex.from_documents(documents=documents,
                                                            transformations=[self.text_splitter,embed_model])
    def ask(self, query: str):
        # Create a retriever from the vector index
        if self.vector_index==None: 
            return "No documents have been ingested yet. Please ingest a document first."
        retriever = self.vector_index.as_retriever(similarity_top_k=3)
        #retrieve the index from query
        nodes= retriever.retrieve(query)
        context=''.join([node[0] for node in nodes])
        # insert query to format the template
        self.prompt = self.qa_template.format(llm=self.llm,query_str=query,context_str=context)
        return self.prompt


    def clear(self):
        # Clear the retriever and pipeline
        self.vector_index = None
        self.retriever = None
        self.prompt=None
