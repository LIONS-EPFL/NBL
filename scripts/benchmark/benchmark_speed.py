import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

#from awq import AutoAWQForCausalLM, BaseAWQForCausalLM
from awq import AutoAWQForCausalLM


class TimeMeasuringLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.token_times = [time.time()]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """The logit processor is called after the model forward."""

        # cuda runs async operates, so we synchronize for accurate time measurement
        torch.cuda.synchronize()

        # measure time
        start_time = time.time()
        self.token_times.append(start_time)
        return scores

    def get_prefill_duration(self):
        return self.token_times[1] - self.token_times[0]

    def get_decode_durations(self):
        token_times = self.token_times[1:]
        token_durations = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]

        return token_durations

def warmup(model):
    warm_up = torch.randn((4096, 4096)).to(next(model.parameters()).device)
    torch.mm(warm_up, warm_up)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def generate_torch(model, input_ids, n_generate):
    context_time = 0
    generate_time = []

    with torch.inference_mode():
        for i in range(n_generate):
            torch.cuda.synchronize()
            start = time.time()

            if i == 0:
                inputs = torch.as_tensor(input_ids, device=next(model.parameters()).device)
            else:
                inputs = torch.as_tensor(token, device=next(model.parameters()).device)

            out = model(inputs, use_cache=True)

            torch.cuda.synchronize()
            token = out[0][:, -1].max(1)[1].unsqueeze(1)

            if i == 0:
                context_time += time.time() - start
            else:
                generate_time.append(time.time() - start)
    
    #print(1/np.sort(generate_time))

    return context_time, generate_time


def load_model(model_path, model_type, quant_file, n_generate, batch_size, no_safetensors, pretrained, model=None):
    print(f" -- Loading model...")

    if model_type == "normal":
        if model is not None:  # use the last loaded model to save time
            return model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    elif model_type == "quantized":
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            torch_dtype=torch.float16,
            fuse_layers=False,
        )

    else:
        raise ValueError(model_type)

    return model


def run_round(generator, model, n_generate, input_ids, batch_size, pretrained):
    model.eval()
    total_memory_used = 0
    for device in range(torch.cuda.device_count()):
        memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        total_memory_used += memory_used
        memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100
        print(f" ** Max Memory (device: {device}): {memory_used:.4f} GB ({memory_pct:.4f}%)")

    print(f"Memory (VRAM): {total_memory_used:.4f} GB ({memory_pct:.4f}%)")

    print(f" -- Warming up...")
    warmup(model)
    #warmup_model(model, 124256, input_ids.shape[1])

    print(f" -- Generating {n_generate} tokens, {input_ids.shape[1]} in context...")

    try:
        context_time, generate_time = generator(model, input_ids, n_generate)
        successful_generate = True
    except RuntimeError as ex:
        if 'cuda out of memory' in str(ex).lower():
            successful_generate = False
        else:
            raise RuntimeError(ex)

    total_memory_used = 0
    memory_pct = 100
    if successful_generate:
        # number of tokens in context / time for processing context * batch size
        prefill_tokens_per_second = round(input_ids.shape[1] / context_time * batch_size, 2)
        # 1 second / median time per token in seconds * batch size
        decode_tokens_per_second = round(1 / np.median(generate_time) * batch_size, 2)

        print(f" ** Speed (Prefill): {prefill_tokens_per_second:.4f} tokens/second")
        print(f" ** Speed (Decode): {decode_tokens_per_second:.4f} tokens/second")

        for device in range(torch.cuda.device_count()):
            memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            total_memory_used += memory_used
            memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100
            print(f" ** Max Memory (device: {device}): {memory_used:.2f} GB ({memory_pct:.4f}%)")
    else:
        prefill_tokens_per_second = 'OOM'
        decode_tokens_per_second = 'OOM'

    if pretrained:
        version = "FP16"
    else:
        try:
            version = model.quant_config.version
        except:
            version = "gptq"
    return {
        "Batch Size": batch_size,
        "Prefill Length": input_ids.shape[1],
        "Decode Length": n_generate,
        "Prefill tokens/s": prefill_tokens_per_second,
        "Decode tokens/s": decode_tokens_per_second,
        "Memory (VRAM)": f"{total_memory_used:.2f} GB ({memory_pct:.2f}%)"
    }, version


def main(args):
    rounds = [
        {"context": 2048, "n_generate": 2048},
    ]

    if args.generator == "torch":
        generator = generate_torch
    else:
        raise ValueError(f"Unknown generator method passed: {args.generator}")

    all_stats = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = None
    for settings in rounds:
        input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, settings["context"])).cuda()

        model = load_model(
            args.model_path,
            args.model_type,
            args.quant_file,
            settings["n_generate"],
            args.batch_size,
            args.no_safetensors,
            args.pretrained,
            model=model
        )

        stats, model_version = run_round(
            generator,
            model,
            settings["n_generate"],
            input_ids,
            args.batch_size,
            args.pretrained
        )

        all_stats.append(stats)

        if stats["Prefill tokens/s"] == 'OOM':
            break

    df = pd.DataFrame(all_stats)
    if args.save_file is not None:
        create_dir(os.path.dirname(args.save_file))
        df.to_csv(args.save_file, index=False)
        print(f"Results saved to \"{args.save_file}\"!")
    print('GPU:', torch.cuda.get_device_name())
    print('Model:', args.model_path)
    print('Version:', model_version)
    print(df.to_markdown(index=False))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-v0.1", help="path to the model")
    parser.add_argument("--model_type", type=str, default="quantized", choices=["normal", "quantized"], help="the type of the model")
    parser.add_argument("--save_file", type=str, default=None, help="path to save the results")
    parser.add_argument("--quant_file", type=str, default="", help="weights filename")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for cache and generation")
    parser.add_argument("--no_safetensors", default=False, action="store_true", help="Use for disabling safetensors")
    parser.add_argument("--generator", type=str, default="torch", choices=["torch", "hf"], help="weights filename")
    parser.add_argument("--pretrained", default=False, action="store_true", help="Measure pretrained model.")
    args = parser.parse_args()

    main(args)