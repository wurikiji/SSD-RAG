import fire
import time
import os
import chromadb
import torch
import json
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM
import torch.nn.functional as F

def get_chroma_client(dir: str):
    chroma_client = chromadb.PersistentClient(path=dir)
    return chroma_client.get_or_create_collection(name="doc_collection")

class Document:
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text

def parse_json_query(json_query: str):
    parsed = json.loads(json_query)
    query = parsed['query']
    return query

def chunked(lst, batch_size: int):
    """Helper to yield successive batch-sized chunks from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

class QueryProcessor():
    def __init__(
        self, 
        query_file: str,
        db_dir: str, 
        cache_dir: str,
        top_k: int = 4,
        model_name: str = "meta-llama/Llama-3.1-8B", 
        use_past_cache: bool = True,
        batch_size: int = 1,
    ):
        self.query_file = query_file
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.use_past_cache = use_past_cache
        self.batch_size = batch_size
        self.vectordb = get_chroma_client(db_dir)
        print("Load model", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 왼쪽 패딩을 사용하도록 설정
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            use_flash_attention_2=True,
        )
        print("Model loaded", flush=True)

    def process_query(self):
        with open(self.query_file) as f:
            lines = f.readlines()

        # 각 줄마다 query 추출
        queries = [parse_json_query(line) for line in lines]
        total_elapsed = 0.0
        total_cache_elapsed = 0.0

        # 배치 단위로 처리
        for batch_queries in chunked(queries, self.batch_size):
            batch_inputs = []      # 각 배치 내 입력 문자열
            batch_caches = []      # 각 배치 내 KV 캐시 (문서 별 concat된 캐시)
            
            # 배치 내 각 쿼리에 대해 top_k 문서 검색 및 캐시 로드
            for query in batch_queries:
                top_k_docs = self.find_top_k_docs(query)
                input_str = self.concatenate_query_and_doc(top_k_docs, query)
                batch_inputs.append(input_str)
                if self.use_past_cache:
                    caches = self.load_all_caches(top_k_docs)
                    concatenated = self.concat_caches(caches)
                    batch_caches.append(concatenated)

            # 배치 토큰화 (left padding)
            tokenized = self.tokenizer(batch_inputs, return_tensors="pt", padding=True).to("cuda")
            
            # 배치 내 각 쿼리의 KV 캐시 패딩 (각 레이어별 최대 시퀀스 길이에 맞춰 왼쪽 패딩)
            if self.use_past_cache:
                cache_load_start = time.perf_counter()
                padded_cache = self.pad_caches_batch(batch_caches)
                # legacy 캐시를 DynamicCache 형식으로 변환
                past_kv_cache = DynamicCache.from_legacy_cache(padded_cache)
                cache_load_end = time.perf_counter()
                cache_interval = cache_load_end - cache_load_start
                print(f"Batch cache load {cache_interval} seconds", flush=True)
                total_cache_elapsed += cache_interval
            else:
                past_kv_cache = None

            # 배치 단위 최초 토큰 생성
            start = time.perf_counter()
            self.generate_first_token_batch(tokenized, past_key_values=past_kv_cache)
            end = time.perf_counter()
            batch_interval = end - start
            print(f"Batch of {len(batch_queries)} queries took {batch_interval} seconds", flush=True)
            total_elapsed += batch_interval

        num_queries = len(queries)
        print(f"Avg elapsed time per query: {total_elapsed / num_queries} seconds", flush=True)
        if self.use_past_cache:
            print(f"Avg cache load time per query: {total_cache_elapsed / num_queries} seconds", flush=True)

    def find_top_k_docs(self, query: str):
        '''
        Select top k documents from the vector db.
        '''
        print(f"Find top {self.top_k} docs for query: {query}", flush=True)
        outputs = self.vectordb.query(query_texts=[query], n_results=self.top_k)
        ids = outputs['ids'][0]
        documents = outputs["documents"][0]

        docs: List[Document] = []
        for id, text in zip(ids, documents):
            docs.append(Document(id, text))
        
        return docs

    def concatenate_query_and_doc(self, docs: List[Document], query: str):
        # 문서 텍스트를 이어 붙이고 쿼리 추가
        input_str = ""
        for doc in docs:
            input_str += doc.text
        input_str += f"\n\nQuestion: {query}"
        return input_str

    def load_all_caches(self, docs: List[Document]):
        '''
        Load past KV caches from disk for all documents.
        '''
        print("Load all caches for docs", flush=True)
        caches = [self.load_kv_cache(doc.id) for doc in docs]
        return caches

    def load_kv_cache(self, doc_id: str):
        '''
        Load past KV cache from disk.
        '''
        cache_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        cache = torch.load(cache_file, map_location="cuda:0", weights_only=True)
        return cache

    def concat_caches(self, caches: List):
        '''
        Concatenate caches for a single query (from its top_k docs).
        각 캐시의 shape은 (1, num_heads, seq_len, head_dim)이고, seq_len은 문서마다 다를 수 있음.
        '''
        if len(caches) == 0:
            return None
        print(f"Concat {len(caches)} caches", flush=True)
        num_layers = len(caches[0])
        concatenated = []
        for layer in range(num_layers):
            # doc별 캐시의 key와 value를 시퀀스 차원(dim=2)으로 cat
            keys = torch.cat([cache[layer][0] for cache in caches], dim=2)
            values = torch.cat([cache[layer][1] for cache in caches], dim=2)
            concatenated.append((keys, values))
        return concatenated

    def pad_caches_batch(self, caches_batch: List[List[tuple]]):
        '''
        배치 내 각 쿼리의 캐시를, 레이어별로 최대 시퀀스 길이에 맞춰 왼쪽 패딩합니다.
        
        :param caches_batch: 배치 내 각 쿼리의 캐시 리스트. 
                             각 요소는 num_layers 길이의 리스트이며, 각 원소는 (keys, values) 튜플.
                             keys, values의 shape: (1, num_heads, seq_len, head_dim)
        :return: 레이어별로 배치 차원이 추가된 캐시 리스트.
                 각 튜플의 shape: (batch_size, num_heads, max_seq_len, head_dim)
        '''
        num_layers = len(caches_batch[0])
        batched_cache = []
        for layer in range(num_layers):
            keys_list = []
            values_list = []
            # 배치 내 해당 레이어의 최대 seq_len 계산
            max_seq_len = max(item[layer][0].shape[2] for item in caches_batch)
            for cache in caches_batch:
                key, value = cache[layer]
                seq_len = key.shape[2]
                pad_len = max_seq_len - seq_len
                if pad_len > 0:
                    # pad: (pad_left, pad_right, pad_top, pad_bottom) -> 여기서는 시퀀스 dim (dim=2)에 대해 왼쪽(pad_top) 패딩
                    key = F.pad(key, (0, 0, pad_len, 0))
                    value = F.pad(value, (0, 0, pad_len, 0))
                keys_list.append(key)    # 각 key: (1, num_heads, max_seq_len, head_dim)
                values_list.append(value)
            # 배치 차원으로 합치기 (각각 (batch_size, num_heads, max_seq_len, head_dim))
            batched_keys = torch.cat(keys_list, dim=0)
            batched_values = torch.cat(values_list, dim=0)
            batched_cache.append((batched_keys, batched_values))
        return batched_cache

    def generate_first_token_batch(self, tokens, past_key_values=None):
        print("Generate first token for batch", flush=True)
        with torch.no_grad():
            self.model.generate(
                **tokens, 
                use_cache=True, 
                past_key_values=past_key_values,
                max_new_tokens=1,
            )

def main(
    query_file: str,
    db_dir: str, 
    cache_dir: str,
    top_k: int = 4, 
    use_past_cache: bool = True,
    batch_size: int = 1,
):
    processor = QueryProcessor(
        query_file=query_file,
        db_dir=db_dir,
        cache_dir=cache_dir,
        top_k=top_k,
        use_past_cache=use_past_cache,
        batch_size=batch_size,
    )

    processor.process_query()

if __name__ == "__main__":
    fire.Fire(main)
