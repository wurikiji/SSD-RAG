import fire
import os
import chromadb
from typing import List
from transformers import LlamaTokenizer, LlamaForCausalLM


def get_chroma_client(dir: str):
  chroma_client = chromadb.PersistentClient(path = dir)
  return chroma_client.get_or_create_collection(name = "doc_collection")

class DocumentChunk():
  def __init__(
    self,
    id: str,
    text: str, 
    tokens,
  ):
    self.id = id
    self.text = text
    self.tokens = tokens


class DocumentPreprocessor():
  def __init__(
      self, 
      ckpt_dir: str,
      docs_dir: str,
      db_dir: str, 
      cache_dir: str,
      chunk_size: int = 512,
  ):
    self.docs_dir = docs_dir
    self.cache_dir = cache_dir
    self.chunk_size = chunk_size
    self.vectordb = get_chroma_client(db_dir)
    model_name = "meta-llama/Llama-2-7b-hf"
    self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
    self.model = LlamaForCausalLM.from_pretrained(model_name)

  def process_documents(self):
    for filename in os.listdir(self.docs_dir):
      chunks = self.split_document(filename)
      self.save_to_vectordb(chunks)

  def split_document(
      self,
      filename: str, 
  ):
    with open(os.path.join(self.docs_dir, filename)) as f:
      text = f.read()
      tokens = self.tokenizer.encode(text, add_special_tokens=False)
      chunks = [
        DocumentChunk(
          id=f"filename-{i}",
          tokens=tokens[i:i+self.chunk_size],
          text=self.tokenizer.decode(
            tokens[i:i+self.chunk_size], skip_special_tokens=True
          )
        ) for i in range(0, len(tokens), self.chunk_size)
      ]
      return chunks

  def save_to_vectordb(self, chunks: List[DocumentChunk]):
    self.vectordb.upsert(
      documents=[chunk.text for chunk in chunks],
      ids=[chunk.id for chunk in chunks]
    )

  def save_kv_cache(self, chunks: List[DocumentChunk]):
    outputs = 

def main(
    ckpt_dir: str,
    docs_dir: str,
    db_dir: str, 
    cache_dir: str,
    chunk_size: int = 512,
):
    preprocessor = DocumentPreprocessor(
      ckpt_dir=ckpt_dir,
      docs_dir=docs_dir,
      db_dir=db_dir,
      cache_dir=cache_dir,
      chunk_size=chunk_size
    )

    preprocessor.process_documents()




if __name__ == "__main__":
  fire.Fire(main)