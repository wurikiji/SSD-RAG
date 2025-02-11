import fire
import time
import os
import chromadb
import torch
import json
from typing import List
from threading import Timer, Lock
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM


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


class QueryProcessor:
    def __init__(
        self, 
        query_file: str,
        db_dir: str, 
        cache_dir: str,
        top_k: int = 4,
        model_name: str = "meta-llama/Llama-3.1-8B", 
        use_past_cache: bool = True,
    ):
        self.query_file = query_file
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.use_past_cache = use_past_cache
        self.vectordb = get_chroma_client(db_dir)
        print("Load model", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            use_flash_attention_2=True,
        )
        print("Model loaded", flush=True)
        # 쿼리 수행 시간을 저장할 리스트와 lock 초기화
        self.query_times: List[float] = []
        self.lock = Lock()

    def process_query_line(self, line: str):
        """
        파일의 한 줄(하나의 쿼리)을 처리하는 함수.
        """
        start = time.perf_counter()
        query = parse_json_query(line)
        top_k_docs = self.find_top_k_docs(query)
        input_text = self.concatenate_query_and_doc(top_k_docs, query)

        if self.use_past_cache:
            cache_load_start = time.perf_counter()
            caches = self.load_all_caches(top_k_docs)
            cache_load_end = time.perf_counter()
            cache_interval = cache_load_end - cache_load_start
            print(f"cache load {cache_interval:.4f} seconds", flush=True)
            concatenated = self.concat_caches(caches)
            self.generate_first_token(input_text, cache=concatenated)
        else:
            self.generate_first_token(input_text)
        end = time.perf_counter()
        total_interval = end - start
        print(f"Query processed in {total_interval:.4f} seconds", flush=True)
        # 실행 시간 저장 (스레드 안전하게)
        with self.lock:
            self.query_times.append(total_interval)

    def process_queries_concurrently(self, rate: float):
        """
        파일 내의 모든 쿼리를 초당 rate 회 실행하도록 Timer를 통해 예약합니다.
        각 쿼리는 독립적인 스레드에서 실행되며, 모든 쿼리가 완료된 후 평균 실행 시간을 출력합니다.
        """
        with open(self.query_file) as f:
            lines = f.readlines()
        print(f"Total queries: {len(lines)}", flush=True)
        timers = []
        for i, line in enumerate(lines):
            delay = i / rate  # i번째 쿼리는 i/rate 초 후에 실행
            timer = Timer(delay, self.process_query_line, args=(line,))
            timer.start()
            timers.append(timer)
            print(f"Scheduled query {i+1} to run after {delay:.4f} seconds", flush=True)
        
        # 모든 Timer 스레드가 종료될 때까지 대기
        for timer in timers:
            timer.join()
        
        # 모든 쿼리가 완료된 후 평균 실행 시간 계산
        if self.query_times:
            avg_time = sum(self.query_times) / len(self.query_times)
            print(f"Avg query execution time: {avg_time:.4f} seconds", flush=True)
        else:
            print("No queries were processed.", flush=True)

    def process_query(self):
        """
        기존의 순차 실행 방식 (모든 쿼리를 순서대로 처리)
        """
        with open(self.query_file) as f:
            lines = f.readlines()

        elapsed = 0.0
        for line in lines:
            query = parse_json_query(line)
            top_k_docs = self.find_top_k_docs(query)
            input_text = self.concatenate_query_and_doc(top_k_docs, query)

            start = time.perf_counter()
            if self.use_past_cache:
                cache_load_start = time.perf_counter()
                caches = self.load_all_caches(top_k_docs)
                cache_load_interval = time.perf_counter() - cache_load_start
                print(f"cache load {cache_load_interval:.4f} seconds", flush=True)
                concatenated = self.concat_caches(caches)
                self.generate_first_token(input_text, cache=concatenated)
            else:
                self.generate_first_token(input_text)
            end = time.perf_counter()
            interval = end - start
            print(f"{interval:.4f} seconds", flush=True)
            elapsed += interval
        print(f"Avg elapsed time: {elapsed / len(lines):.4f} seconds", flush=True)

    def find_top_k_docs(self, query: str):
        '''
        select top k documents from the vector db
        '''
        print(f"Find top {self.top_k}", flush=True)
        outputs = self.vectordb.query(query_texts=[query], n_results=self.top_k)
        ids = outputs['ids'][0]
        documents = outputs["documents"][0]

        docs: List[Document] = []
        for id, text in zip(ids, documents):
            docs.append(Document(id, text))
        return docs

    def concatenate_query_and_doc(self, docs: List[Document], query: str):
        input_text = ""
        for doc in docs:
            input_text += doc.text
        input_text += f"\n\nQuestion: {query}\nAnswer: "
        return input_text

    def load_all_caches(self, docs: List[Document]):
        '''
        load past kv cache from the disk for all documents
        '''
        print("Load all caches", flush=True)
        caches = [self.load_kv_cache(doc.id) for doc in docs]
        return caches

    def load_kv_cache(self, doc_id: str):
        '''
        load past kv cache from the disk
        '''
        cache_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        cache = torch.load(cache_file)
        return cache

    def concat_caches(self, caches):
        '''
        concatenate the cache 
        '''
        if len(caches) == 0:
            return None
        print(f"Concat {len(caches)} caches", flush=True)
        num_layers = len(caches[0])
        concatenated = []
        for layer in range(num_layers):
            keys = torch.cat([cache[layer][0] for cache in caches], dim=2)
            values = torch.cat([cache[layer][1] for cache in caches], dim=2)
            concatenated.append((keys, values))
        return concatenated

    def generate_first_token(self, input_text: str, cache=None):
        print("Generate first token", flush=True)
        if cache is not None:
            past_kv_cache = DynamicCache.from_legacy_cache(cache)
        else:
            past_kv_cache = None
        token = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **token, 
                use_cache=True, 
                past_key_values=past_kv_cache,
                max_new_tokens=100,
                temperature=0.01
            )
            input_length = token["input_ids"].shape[1]
            output_string = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
            print("==========INPUT========", flush=True)
            print(input_text, flush=True)
            print("==========ANSWER=========", flush=True)
            print(output_string, flush=True)


def main(
    query_file: str,
    db_dir: str, 
    cache_dir: str,
    top_k: int = 4, 
    use_past_cache: bool = True,
    rate: float = 0  # rate > 0: 초당 실행 횟수, 0이면 순차 처리
):
    processor = QueryProcessor(
        query_file=query_file,
        db_dir=db_dir,
        cache_dir=cache_dir,
        top_k=top_k,
        use_past_cache=use_past_cache,
    )

    if rate > 0:
        print(f"Processing queries concurrently at {rate} per second", flush=True)
        processor.process_queries_concurrently(rate)
    else:
        print("Processing queries sequentially", flush=True)
        processor.process_query()


if __name__ == "__main__":
    fire.Fire(main)
