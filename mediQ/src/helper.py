import torch
import logging
from keys import mykey

# A dictionary to cache models and tokenizers to avoid reloading

global models
models = {}

def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")

class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.api_account = self.args.get("api_account", "mediQ")
        self.api_base_url = self.args.get("api_base_url", None)
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api in {"openai", "deepseek"}:
            from openai import OpenAI
            if self.api_account not in mykey:
                raise KeyError(f"API account '{self.api_account}' not found in keys.py")
            client_kwargs = {"api_key": mykey[self.api_account]}
            base_url = self.api_base_url
            if self.use_api == "deepseek":
                client_kwargs["base_url"] = base_url or "https://api.deepseek.com"
            elif base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)  # Setup API client appropriately in keys.py
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                self.model = LLM(model=self.model_name, enable_prefix_caching=enable_prefix_caching)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            except Exception as e:
                log_info(f"[ERROR] [{self.model_name}]: If using a custom local model, it is not compatible with VLLM, will load using Huggingfcae and you can ignore this error: {str(e)}", mode="error")
                self.use_vllm = False
        if not self.use_vllm and self.use_api not in {"openai", "deepseek"}:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    def generate(self, messages):
        log_info(f"[{self.model_name}][INPUT]: {messages}")

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_length", self.args.get("max_tokens", 256))
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if self.use_api == "openai":
            return self.openai_generate(messages)
        if self.use_api == "deepseek":
            return self.deepseek_generate(messages)
        if self.use_vllm:
            return self.vllm_generate(messages)
        return self.huggingface_generate(messages)
    
    def huggingface_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            prompt = "\n\n".join([m['content'] for m in messages])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=self.max_tokens, 
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.terminators
        )
        # TODO: If top_logprobs > 0, return logprobs of generation
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        usage = {"input_tokens": inputs.shape[-1], "output_tokens": outputs.shape[-1]-inputs.shape[-1]}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage
        
    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.top_logprobs > 0:
            params["logprobs"] = True
            params["top_logprobs"] = self.top_logprobs
        response = self.client.chat.completions.create(**params)
        response_text, log_probs = self._extract_completion(response)
        usage = self._extract_usage(response)
        log_info(f"[{self.model_name}][OUTPUT]: {response}")
        return response_text, log_probs, usage

    def deepseek_generate(self, messages):
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.top_logprobs > 0:
            log_info(f"[{self.model_name}] DeepSeek API does not support logprobs; ignoring top_logprobs={self.top_logprobs}", mode="warning")
        response = self.client.chat.completions.create(**params)
        response_text, _ = self._extract_completion(response)
        usage = self._extract_usage(response)
        log_info(f"[{self.model_name}][OUTPUT][DeepSeek]: {response_text}")
        return response_text, None, usage

    def _extract_completion(self, response):
        choice = response.choices[0]
        if hasattr(choice, "message") and getattr(choice.message, "content", None):
            response_text = choice.message.content.strip()
        else:
            response_text = getattr(choice, "text", "").strip()
        log_probs = getattr(choice, "logprobs", None)
        return response_text, log_probs

    def _extract_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0}
        return {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0)
        }


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    if 'gpt' in model_name or 'o1' in model_name: use_api = "openai"
    
    model_cache = models.get(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[model_name] = model_cache
    print(f"[learning process] [helper.py line 157] printout the messages {type(messages)}, content: {messages}")
    return model_cache.generate(messages)
