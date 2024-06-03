# ELYZA-japanese-Llama-2-7b-instruct

!pip install auto-gptq transformers

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_name_or_path = "mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-GPTQ-calib-ja-2k"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# Model
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, use_safetensors=True, device="cuda:0", use_auth_token=False)

#Your test prompt
prompt = """[INST] <<SYS>>あなたは誠実で優秀な日本人のアシスタントです。<</SYS>>クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。 [/INST]"""
print(tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=512)[0]))

# Phi-3-mini-128k-instruct
!pip install accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

# RAGお試し
!pip install openai==0.28.1
!pip install transformers[torch]==4.34.1 accelerate==0.24.0 InstructorEmbedding==1.0.1 sentence_transformers==2.2.2 pypdf==3.16.4 optimum==1.13.2 auto-gptq==0.4.2 llama-index==0.9.8

from typing import Any, List

import torch
from InstructorEmbedding import INSTRUCTOR
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

model_name = "TheBloke/Xwin-LM-7B-V0.1-GPTQ"

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    tokenizer_name=model_name,
    model_name=model_name,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16},
    generate_kwargs={"temperature": 0.01, "do_sample": True},
)

class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _query_prefix: str = PrivateAttr()
    _text_prefix: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = 'intfloat/multilingual-e5-large',
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._query_prefix = "query:" # multilingual-e5-largeの学習方法に準拠: https://huggingface.co/intfloat/multilingual-e5-large
        self._text_prefix = "passage:" # multilingual-e5-largeの学習方法に準拠: https://huggingface.co/intfloat/multilingual-e5-large
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._query_prefix, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._text_prefix, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([[self._text_prefix, text] for text in texts])
        return embeddings

# ベクトル化モジュールの設定
service_context = ServiceContext.from_defaults(llm=llm,
    embed_model=InstructorEmbeddings(embed_batch_size=1), chunk_size=512
)
# ドキュメントをgithubからダウンロード
!mkdir -p data
!wget -P data https://github.com/nyanta012/open-model-tutorial/raw/main/pdf/健康のすべて.pdf

# ドキュメント読み込み
documents = SimpleDirectoryReader("data").load_data()

# インデックスの作成
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

TEMPLATE ="""A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: 下記の情報が与えられています。
\n ---------------------\n {context_str} \n---------------------\n
この情報を参照して次の質問に答えてください。情報に基づいた回答のみ生成してください。
情報にない場合は、わからない旨を伝えてください。
質問:{query_str} ASSISTANT:"""

PROMPT = PromptTemplate(TEMPLATE)

query_engine = index.as_query_engine(text_qa_template=PROMPT, streaming=True)

response = query_engine.query("健康になるにはどんな運動をすると良いですか？")
response.print_response_stream()

print((response.source_nodes[0].node.get_text()))

print((response.source_nodes[1].node.get_text()))
