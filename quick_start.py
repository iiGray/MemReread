from transformers import AutoTokenizer
import argparse
from exp.interface.mem_reread_api import MemReread_API
from exp.tools import *


def chunk_worker(item, model, tokenizer):
    context_ids = tokenizer.encode(item['context'])
    tokenized_chunks = [context_ids[i: i + 5000] for i in range(0, len(context_ids), 5000)]
    item['context_chunks'] = [tokenizer.decode(ck) for ck in tokenized_chunks]
    return item


async def main(args):
    agent = MemReread_API(
        tokenizer_model = args.tokenizer_model,
        model_name = args.model,
        base_url = args.base_url,
        api_key = args.api_key,
        max_width = args.max_reread + 1,
        max_nodes = args.max_reread + 1
    )

    data = read_json("./datas/quick_start_data.json")
    
    assert data['question'] and data['context'], "Please set your question & context in `./datas/quick_start_data.json`"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    data = chunk_worker(data, model = args.model, tokenizer = tokenizer)
    result, recorded = await agent(
        question = data['question'],
        context = data['context'],
        context_chunks = data['context_chunks']
    )
    save_json(dict(result = result, recorded = recorded), "./datas/quick_start_response.json")

    print("The result has been save in `./datas/quick_start_response.json` successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-model", "-tm", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--base-url", "-bs", type=str, default="")
    parser.add_argument("--api-key", "-ak", type=str, default="")
    parser.add_argument("--max-reread", "-mr", type = int, default = 3)
    args = parser.parse_args()
    async_run(main(args))