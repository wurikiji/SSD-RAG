import fire
import time
import os
import chromadb
import torch
import json
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM


def get_chroma_client(dir: str):
  chroma_client = chromadb.PersistentClient(path = dir)
  return chroma_client.get_or_create_collection(name = "doc_collection")

class Document:
  def __init__(self, id: str, text: str):
    self.id = id
    self.text = text


def parse_json_query(json: str):
    parsed = json.loads(json)
    query = parsed['query']
    return query

class QueryProcessor():
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
    print("Load model")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      torch_dtype=torch.float16,
      device_map="auto",
      use_flash_attention_2=True,
    )
    print("Model loaded")

  def process_query(self):
    with open(self.query_file) as f:
      lines = f.readlines()
      for line in lines:
          query = parse_json_query(line)
          top_k_docs = self.find_top_k_docs(query)

          start = time.perf_counter()
          if self.use_past_cache:
            caches = self.load_all_caches(top_k_docs)
            concatenated = self.concat_caches(caches)
            self.generate_first_token(concatenated)
          else:
            self.generate_first_token()
          
          end = time.perf_counter()

          print(f"{end - start} seconds")
          
          break
          
  
  def find_top_k_docs(self, query: str):
    '''
    select top k documents from the vector db
    '''
    outputs = self.vectordb.query(query_texts=[query], n_results=self.top_k)
    ids = outputs['ids'][0]
    documents = outputs["documents"][0]

    docs: List[Document] = []
    for id, text in zip(ids, documents):
      docs.append(Document(id, text))
    
    return docs
  
  def load_all_caches(self, docs: List[Document]):
    '''
    load past kv cache from the disk for all documents
    '''
    caches = []
    for doc in docs:
      caches.append(self.load_kv_cache(doc.id))
    
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
    num_layers = len(caches[0])
    concatenated = []
    for layer in range(num_layers):
      keys = torch.cat([cache[layer][0] for cache in caches])
      values = torch.cat([cache[layer][1] for cache in caches])
      concatenated.append((keys, values))
    return concatenated
  
  def generate_first_token(
      self, 
      input: str,
      cache = None,
  ):
    if cache is not None:
      past_kv_cache = DynamicCache.from_legacy_cache(cache)
    else:
      past_kv_cache = DynamicCache()
    token = self.tokenizer(input, return_tensors="pt").to("cuda")
    with torch.no_grad():
      self.model.generate(
        **token, 
        use_cache=True, 
        past_key_values = past_kv_cache,
        max_new_tokens = 1,
      )

def main(
    query_file: str,
    db_dir: str, 
    cache_dir: str,
    top_k: int = 4, 
    use_past_cache: bool = True,
):
    processor = QueryProcessor(
      query_file=query_file,
      db_dir=db_dir,
      cache_dir=cache_dir,
      top_k=top_k,
      use_past_cache=use_past_cache,
    )

    processor.process_query()




if __name__ == "__main__":
  fire.Fire(main)