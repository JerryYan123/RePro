from vllm import LLM, TokensPrompt, SamplingParams
from pathlib import Path as LocalPath
from google.cloud import storage
from cloudpathlib import S3Path
import zstandard as zstd
from tqdm import tqdm
import datasets
import argparse
import json
import re

recycle_prompt = """Your task is to read and paraphrase the provided text following these instructions:
- Delete clearly irrelevant content:
  - Website headers, navigation bars, or menu items (e.g., "Home | About | Contact")
  - Unrelated HTTP links (e.g., ads, trackers, developer tools)
  - Generic footers (e.g., contact info, privacy policies, unsubscribe links)
  - Empty lines or decorative elements (e.g., "---")
- Preserve all content that is relevant and meaningful:
  - Informative or independently useful
  - Related to the topic, even tangentially
  - Provides context, background, or supporting value
  - Includes technical terms, key concepts, factual details, reasoning, and examples
- Handle mixed-relevance sentences carefully:
  - Remove only the irrelevant fragment if the rest remains coherent
  - Delete the whole sentence if the remainder loses meaning
- Do not alter meaningful content unnecessarily:
  - Only delete or modify when content is clearly meaningless or off-topic
  - Preserve the original structure, logic, and depth of the text
- Do not add explanations, notes, assumptions, or claims not found in the original text
Here is the text:
{TEXT}
Task:
After thoroughly reading the above text, paraphrase it in high-quality and clear English following the instructions.
Start your response immediately with "Here is a paraphrased version:" and then provide the paraphrased text."""

gstore = storage.Client().bucket("")
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.9,
    max_tokens=2048,
)


def load_jsonl_zst(file_path):
    """Load data from a compressed JSONL file and yield examples"""
    if file_path.startswith("s3://"):
        path = S3Path(file_path)
    else:
        path = LocalPath(file_path)

    with path.open("rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode("utf-8")
            for line in text_stream.strip().split("\n"):
                if line:
                    yield json.loads(line)


def make_conversation(
    example,
    prompt_column: str = "text",
    tokenizer=None,
    max_prompt_tokens=4096,
):
    prompt = []
    prompt.append(
        {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. /no_think",
        }
    )

    text_content = example[prompt_column]

    # If tokenizer is provided, truncate based on token count
    if tokenizer is not None:
        # First, get the base prompt tokens (without the text)
        base_prompt = recycle_prompt.replace("{TEXT}", "")
        base_tokens = tokenizer.encode(base_prompt)
        system_tokens = tokenizer.encode("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions.")

        # Calculate available tokens for the text content
        available_tokens = (
            max_prompt_tokens - len(base_tokens) - len(system_tokens) - 50
        )  # 50 token safety margin

        # Tokenize and truncate the text content
        text_tokens = tokenizer.encode(text_content)
        if len(text_tokens) > available_tokens:
            text_content = (
                tokenizer.decode(text_tokens[: available_tokens // 2])
                + "... "
                + tokenizer.decode(text_tokens[-(available_tokens // 2) :])
            )

    prompt.append(
        {
            "role": "user",
            "content": recycle_prompt.format(TEXT=text_content),
        }
    )
    return {"prompt": prompt}


def transform(llm, conversations):
    outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
    contents = []
    for output in outputs:
        content = output.outputs[0].text
        # Find the first occurrence that matches the pattern in the string
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")
    return contents


def vllm_inference(llm, dataset):
    batch_size = 1024
    transformed_texts = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference..."):
        batch = dataset[i : i + batch_size]["prompt"]
        transformed_batch = transform(llm, batch)
        transformed_texts.extend(transformed_batch)
    return transformed_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--write_template", type=str, required=True)
    args = parser.parse_args()

    rank = args.rank
    print(f"Received argument: {rank}")

    llm = LLM(model=args.model)

    tokenizer = llm.get_tokenizer() if "OLMo2-1B" in args.model else None
    read_template = "s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10/shard_{}_processed.jsonl.zstd"
    write_template = args.write_template
    for i in tqdm(range(rank, rank + 1), desc="Processing shards..."):
        write_file_name = write_template.format(str(i).zfill(8))
        if gstore.blob(write_file_name).exists():
            print(f"{write_file_name} already exists!")
            continue
        read_file_name = read_template.format(str(i).zfill(8))
        print(f"Processing file: {read_file_name}")
        data_list = []
        tids = []
        tid = 0
        for item in load_jsonl_zst(read_file_name):
            text = item["text"]
            for j in range(0, len(text), 7000):
                data_list.append({"text": text[j : j + 7000]})
                tids.append(tid)
            tid += 1
        dataset = datasets.Dataset.from_list(data_list).map(
            lambda x: make_conversation(x, tokenizer=tokenizer)
        )
        transformed_texts = vllm_inference(llm, dataset)
        with gstore.blob(write_file_name).open("w") as f:
            print(f"Writing to file: {write_file_name}")
            prev_tid = 0
            texts = []
            cnt = 0
            for tid, text in zip(tids, transformed_texts):
                if tid == prev_tid:
                    texts.append(text)
                else:
                    cur_text = " ".join(texts)
                    if cur_text == "":
                        cnt += 1
                    f.write(json.dumps({"text": cur_text}) + "\n")
                    prev_tid = tid
                    texts = [text]
            cur_text = " ".join(texts)
            if cur_text == "":
                cnt += 1
            # Remove leading and trailing "---"
            if cur_text[:3] == "---":
                cur_text = cur_text[3:]
            if cur_text[-3:] == "---":
                cur_text = cur_text[:-3]
            f.write(json.dumps({"text": cur_text.strip()}) + "\n")
            print(f"Total {len(transformed_texts)} items, {cnt} invalid items found.")


if __name__ == "__main__":
    main()
