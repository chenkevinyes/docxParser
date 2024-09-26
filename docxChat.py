from llama_index.llms.ollama import Ollama
from llama_index.core import Settings,StorageContext,load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import ChatPromptTemplate


def color_print(text: str, mode: str = '', fore: str = '', back: str = '',end="") -> None:
    dict_mode = {'d': '0', 'h': '1', 'nb': '22', 'u': '4', 'nu': '24',
                 't': '5', 'nt': '25', 'r': '7', 'nr': '27', '': ''}
    dict_fore = {'k': '30', 'r': '31', 'g': '32', 'y': '33', 'b': '34',
                 'm': '35', 'c': '36', 'w': '37', '': ''}
    dict_back = {'k': '40', 'r': '41', 'g': '42', 'y': '43', 'b': '44',
                 'm': '45', 'c': '46', 'w': '47', '': ''}
    formats = ';'.join([each for each in [
        dict_mode[mode], dict_fore[fore], dict_back[back]] if each])
    print(f'\033[{formats}m{text}\033[0m',end=end)


# qa_prompt_str = (
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question: {query_str}\n"
# )

# refine_prompt_str = (
#     "We have the opportunity to refine the original answer "
#     "(only if needed) with some more context below.\n"
#     "------------\n"
#     "{context_msg}\n"
#     "------------\n"
#     "Given the new context, refine the original answer to better "
#     "answer the question: {query_str}. "
#     "If the context isn't useful, output the original answer again.\n"
#     "Original Answer: {existing_answer}"
# )
# # Text QA Prompt
# chat_text_qa_msgs = [
#     (
#         "system",
#         "Always answer the question, even if the context isn't helpful.",
#     ),
#     ("user", qa_prompt_str),
# ]
# text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# # Refine Prompt
# chat_refine_msgs = [
#     (
#         "system",
#         "Always answer the question, even if the context isn't helpful.",
#     ),
#     ("user", refine_prompt_str),
# ]
# refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)


Settings.llm = Ollama(base_url="http://localhost:11434", model="qwen2:latest", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(
    model_name="shaw/dmeta-embedding-zh"
)
persist_dir = './storage_vectorstore'
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)
chat_engine= index.as_chat_engine(chat_mode='context',streaming=True, 
                                #   refine_template=refine_template
        )
print('chat engine is ready!')
# color_print('chat engine is ready!',fore='r')
while True:
    q = input('问题：')
    if q != '':
        try:
            response = chat_engine.stream_chat(
                q
            )
            # response.is_dummy_stream=True
            
            for token in response.response_gen:
                # print(token, end="")
                color_print(token,fore='g')
            # response.async_response_gen()
            print("\b")
            chat_engine.reset()
            # response = query_engine.query("红楼梦中的主要人物及其关系介绍")
            # response = query_engine.query("请简要介绍供应链数字化平台的技术架构")
            # print(response) 
        except Exception as e:
            print(e)