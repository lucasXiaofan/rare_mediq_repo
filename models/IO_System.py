# Licensed under the MIT license.

import sys

sys.path.append(".")

from typing import List, Dict

try:
    from models.vLLM_API import generate_with_vLLM_model
except:
    pass

# import ipdb; ipdb.set_trace()
try:
    from models.OpenAI_API import generate_n_with_OpenAI_model, generate_with_OpenAI_model
except:
    pass

try:
    from models.Llama_API import generate_n_with_Llama_model
except:
    pass

# DeepSeek API support
try:
    import os
    from openai import OpenAI
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Initialize DeepSeek client
    deepseek_client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )
except Exception as e:
    deepseek_client = None
    print(f"Failed to initialize deepseek_client: {e}")
    pass


class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        if self.api == "together":
            assert tokenizer is None and model is None
        elif self.api == "gpt3.5-turbo":
            assert tokenizer is None and isinstance(model, str)
        elif self.api == "deepseek":
            assert tokenizer is None and isinstance(model, str)
            assert deepseek_client is not None, "DeepSeek client not initialized. Please install openai and set DEEPSEEK_API_KEY"
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0
        self.token_counter = 0
        self.enable_chat_template = args.enable_chat_template

    def generate(self, model_input, max_tokens: int, num_return: int, stop_tokens):
        if isinstance(model_input, str):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                    tokenizer=self.tokenizer,
                    enable_chat_template=self.enable_chat_template
                )
                io_output_list = [o.text for o in vllm_response[0].outputs]
                self.call_counter += 1
                self.token_counter += sum([len(o.token_ids) for o in vllm_response[0].outputs])
            elif self.api == "gpt3.5-turbo":
                # gpt_response = generate_n_with_OpenAI_model(
                #     prompt=model_input,
                #     n=num_return,
                #     model_ckpt=self.model,
                #     max_tokens=max_tokens,
                #     temperature=self.temperature,
                #     top_p=self.top_p,
                #     top_k=self.top_k,
                #     stop=[],
                #     enable_chat_template=self.enable_chat_template
                # )
                gpt_response = generate_with_OpenAI_model(
                    prompt=model_input,
                    n=num_return,
                    model_ckpt=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop=stop_tokens,
                    enable_chat_template=self.enable_chat_template
                )
                io_output_list = gpt_response
                self.call_counter += num_return
                self.token_counter += 0
            elif self.api == "llama":
                gpt_response = generate_n_with_Llama_model(
                    prompt=model_input,
                    n=num_return,
                    model_ckpt=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop=["\n", "Answer"],
                )
                io_output_list = gpt_response
                self.call_counter += num_return
                self.token_counter += 0
            elif self.api == "deepseek":
                # DeepSeek API call
                print(f"[IO_System.py] num of return: {num_return}")
                deepseek_response = self._generate_with_deepseek(
                    prompt=model_input,
                    n=num_return,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    # stop=stop_tokens,
                )
                io_output_list = deepseek_response
                self.call_counter += num_return
                self.token_counter += 0
            elif self.api == "debug":
                io_output_list = ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")
        elif isinstance(model_input, list):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                    tokenizer=self.tokenizer,
                    enable_chat_template=self.enable_chat_template
                )
                io_output_list = [[o.text for o in resp_to_single_input.outputs] for resp_to_single_input in vllm_response]
                self.call_counter += 1
                self.token_counter += sum(
                    [
                        sum([len(o.token_ids) for o in resp_to_single_input.outputs])
                        for resp_to_single_input in vllm_response
                    ]
                )
            elif self.api == "gpt3.5-turbo":
                io_output_list = []
                for input in model_input:
                    gpt_response = generate_with_OpenAI_model(
                        prompt=input,
                        n=num_return,
                        model_ckpt=self.model,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        stop=stop_tokens,
                    )
                    io_output_list.append(gpt_response)
                    self.call_counter += num_return
                    self.token_counter += 0
            elif self.api == "llama":
                io_output_list = []
                for input in model_input:
                    gpt_response = generate_n_with_Llama_model(
                        prompt=input,
                        n=num_return,
                        model_ckpt=self.model,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        stop=["\n", "Answer"],
                    )
                    io_output_list.append(gpt_response)
                    self.call_counter += num_return
                    self.token_counter += 0
            elif self.api == "deepseek":
                # DeepSeek API call for batch inputs
                io_output_list = []
                for input in model_input:
                    deepseek_response = self._generate_with_deepseek(
                        prompt=input,
                        n=num_return,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stop=stop_tokens,
                    )
                    io_output_list.append(deepseek_response)
                    self.call_counter += num_return
                    self.token_counter += 0
            elif self.api == "debug":
                io_output_list = [
                    ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
                    for _ in model_input
                ]
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")

        return io_output_list

    def _generate_with_deepseek(self, prompt, n=1, max_tokens=256, temperature=0.8, top_p=0.95, stop=None):
        """
        Generate text using DeepSeek API.

        Args:
            prompt: Input text prompt
            n: Number of completions to generate
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            List of generated text completions
        """
        import time

        # Parse prompt into messages if it contains ### Instruction: markers (chat template)
        if self.enable_chat_template and "### Instruction:" in prompt and "### Response:" in prompt:
            segments = prompt.split("### Instruction:")
            messages = [
                {"role": "system", "content": segments[0].strip()},
            ]
            for idx in range(1, len(segments)):
                examples = segments[idx].split("### Response:")
                messages.append({
                    "role": "user",
                    "content": examples[0].strip()
                })
                if len(examples) > 1 and len(examples[1].strip()) > 0:
                    messages.append({
                        "role": "assistant",
                        "content": examples[1].strip()
                    })
        else:
            # Default: wrap prompt in system/user messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]

        # DeepSeek API currently only supports n=1; loop to simulate multiple completions.
        if n <= 0:
            return []

        parameters = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n": 1,
        }

        # Add stop sequences if provided
        if stop and len(stop) > 0:
            parameters["stop"] = stop

        def fetch_single_completion():
            delay = 0
            while True:
                if delay > 0:
                    time.sleep(delay)
                try:
                    completion = deepseek_client.chat.completions.create(**dict(parameters))
                    return [
                        completion.choices[idx].message.content
                        for idx in range(len(completion.choices))
                    ]
                except Exception as e:
                    print(f"DeepSeek API Error: {e}")
                    delay = 1 if delay == 0 else delay * 2
                    if delay > 120:
                        delay = 1
                    try:
                        print(f"Will retry after {delay} seconds ...")
                    except Exception:
                        pass

        completions = []
        for _ in range(n):
            completions.extend(fetch_single_completion())

        return completions[:n]
