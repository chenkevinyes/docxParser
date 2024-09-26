from typing import Dict, Optional
import fsspec
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings,StorageContext,node_parser, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser

from llama_index.core import SimpleDirectoryReader,PromptTemplate
from llama_index.core import VectorStoreIndex,SimpleKeywordTableIndex,RAKEKeywordTableIndex,GPTKeywordTableIndex,GPTSimpleKeywordTableIndex
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
# from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter,SentenceSplitter,MetadataAwareTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from  docx_node_parser import docxNodeParser

# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''
Settings.chunk_size = 1024
Settings.chunk_overlap =20
Settings.llm = Ollama(base_url="http://localhost:11434", model="qwen2:latest", request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(
    model_name="shaw/dmeta-embedding-zh"
) 

keyword_extract_template =PromptTemplate( (
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords. 请以中文输出."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"

))
# query_keyword_extract_template = PromptTemplate((
#     "A question is provided below. Given the question, extract up to {max_keywords} "
#     "keywords from the text. Focus on extracting the keywords that we can use "
#     "to best lookup answers to the question. Avoid stopwords.请以中文输出\n"
#     "---------------------\n"
#     "{question}\n"
#     "---------------------\n"
#     "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
# ))

query_keyword_extract_template = PromptTemplate((
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
    "请以中文输出"
))

qa_prompt_template =PromptTemplate((
"请是供应链数字化中心的智能助手，你总是基于本地知识库回答问题. \n"
"---------------------\n"
"{context_str}"
"\n---------------------\n"
"根据上面的信息, 请回答这个问题: {query_str}\n"
))
refine_prompt_template = PromptTemplate((
    "原始问题如下: {query_str}\n"
    "我们提供了下面的答案: {existing_answer}\n"
    "如果上述答案不是中文，请将其转换为中文输出\n"
    
))

# text_splitter = TokenTextSplitter(
#     separator=" ", chunk_size=512, chunk_overlap=128
# )

def getDocKeywordByName(fileName,llm=Settings.llm):
    keywords = llm.predict(
            prompt = PromptTemplate(template= """\
                    {context_str}. Give {keywords} unique keywords for this \
                    document. Format as comma separated. Keywords: .请输出中文"""),   
            context_str=fileName,
            keywords =  3
        )
    return keywords

class CustomExtractor(BaseExtractor):
    def extract(self, nodes):
        metadata_list = [
            {
                "模块": (
                    "采购管理"
                )
            }
            for node in nodes
        ]
        return metadata_list



def main(): 

    extractors = [
        # TitleExtractor(nodes=5, llm=Settings.llm),
        # QuestionsAnsweredExtractor(questions=3, llm=Settings.llm),
        # EntityExtractor(prediction_threshold=0.5),
        # SummaryExtractor(summaries=["prev", "self"], llm=Settings.llm),
        # KeywordExtractor(keywords=10, llm=Settings.llm,
        #                 keyword_extract_template = """\
        #                 {context_str}. Give {keywords} unique keywords for this \
        #                 document. Format as comma separated. Keywords: .请输出中文"""),
        # CustomExtractor(),
    ]


    documents = SimpleDirectoryReader("./data",required_exts=['.docx'],recursive=True).load_data()
    for item in documents:
        # task = [result(url) for url in url_list]
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(asyncio.wait(task))
        keywords =getDocKeywordByName(item.metadata['file_name'])
        item.metadata['keywords'] = keywords
    parser = docxNodeParser()
    transformations = [parser] + extractors 

    # transformations = extractors

    pipeline = IngestionPipeline(transformations=transformations)

    nodes = pipeline.run(documents=documents)


    # nodes = parser.get_nodes_from_documents(documents)

    with open('./temp/nodes.text',mode='w',encoding='utf-8') as nodesFile:
        for idx, node in enumerate(nodes):
            # node  = query_index.docstore.get_node(nodeId)
            hintCnt = 150 if len(node.text) >300 else int(len(node.text) /2) 
            hintStr = node.text[0:hintCnt]+'......'+node.text[-hintCnt:]            
            # hintStr = node.text
            nodesFile.write(str(idx) + '\t'+str(len(node.text)) + '\t'+node.metadata['file_name']+'\t'+repr(hintStr)+'\n')
            # print(repr(hintStr),'\n')
        nodesFile.close()
    vector_index = VectorStoreIndex(nodes)
    persist_dir = './storage_vectorstore'
    vector_index.storage_context.persist(persist_dir=persist_dir)
main()