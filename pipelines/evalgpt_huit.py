import os
import sys
import re
import json
import time
import warnings
import requests
from dataclasses import dataclass, field
import itertools
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import HfApi
import tiktoken
from openai import AsyncAzureOpenAI, RateLimitError
from aiolimiter import AsyncLimiter
import asyncio
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser
from typing import Any, NamedTuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS, sample_every_k_batched


HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
GPT_CONFIGS = CONFIGS.evaluations.gpt
OPENAI_CONFIGS = CONFIGS.services.openai
CACHE_CONFIGS = CONFIGS.utils.cache
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# Global cost tracking
COST = 0
# Global retry tracking
EXTRA_RETRIES = 0


class WrappedResponse(NamedTuple):
    content: str


class HUITOpenAI:
    """
    Custom chat model for a HUIT OpenAI endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 60,
        api_key: str = None,
    ):
        metadata = {}
        metadata["endpoint_url"] = (
            "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions"
        )
        # Use provided API key, or try to get from environment variable
        if api_key is None:
            api_key = os.getenv("HUIT_API_KEY")
            if api_key is None:
                raise ValueError("API key must be provided either as parameter or HUIT_API_KEY environment variable")
        metadata["api_key"] = api_key
        self.model = model.replace("-huit", "")
        self.metadata = metadata
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts

    def generate_content(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> WrappedResponse:
        # 1. Construct the payload
        payload_dict = {
            "model": self.model,
            "messages": [
                # {"role": "system", "content": extra},
                {"role": "user", "content": prompt}
            ],
        }
        if not self.model.startswith("o3"):
            payload_dict["max_tokens"] = max_tokens
            payload_dict["temperature"] = temperature
            payload_dict["top_p"] = top_p

        payload = json.dumps(payload_dict)

        headers = {
            "Content-Type": "application/json",
            "api-key": self.metadata["api_key"],
        }

        # 2. Send the request
        attempts = 0
        global EXTRA_RETRIES
        while True:
            attempts += 1
            if attempts > self.max_attempts:
                raise RuntimeError("Failed to get a response from the endpoint.")
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                )
                response.raise_for_status()
                if attempts > 1:
                    EXTRA_RETRIES += 1
                    print(f"Extra retry #{EXTRA_RETRIES} for HUIT API (attempt {attempts}/{self.max_attempts})")
                break
            except requests.exceptions.HTTPError as e:
                warnings.warn(f"Attempt {attempts} failed: {e}")
                if attempts > 1:
                    EXTRA_RETRIES += 1
                    print(f"Extra retry #{EXTRA_RETRIES} for HUIT API HTTP error (attempt {attempts}/{self.max_attempts}): {e}")
                time.sleep(self.wait_time_between_attempts)

        # 3. Parse the response
        result_json = response.json()
        global COST
        COST += result_json['your_cost_this_transaction']
        # print(result_json)
        content = result_json["choices"][0]["message"]["content"]

        return WrappedResponse(content)


class AsyncHUITOpenAI:
    """
    Async wrapper for HUITOpenAI to maintain compatibility with the existing async pipeline.
    """
    
    def __init__(self, huit_client: HUITOpenAI):
        self.huit_client = huit_client
    
    async def chat_completions_create(self, **kwargs):
        # Extract messages and convert to a single prompt
        messages = kwargs.get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            # Handle both single message and list of messages
            if isinstance(messages[0], list):
                # This is the case where we have a list of message lists
                # We'll process each message list separately
                results = []
                for message_list in messages:
                    prompt = self._messages_to_prompt(message_list)
                    result = await self._call_huit_async(prompt, kwargs)
                    results.append(result)
                return results
            else:
                # Single message list
                prompt = self._messages_to_prompt(messages)
                result = await self._call_huit_async(prompt, kwargs)
                return result
        else:
            raise ValueError("No messages provided")
    
    def _messages_to_prompt(self, messages):
        """Convert OpenAI-style messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)
    
    async def _call_huit_async(self, prompt, kwargs):
        """Make the actual call to HUITOpenAI in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.huit_client.generate_content(
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
            )
        )
        
        # Return a mock response object that mimics OpenAI's response structure
        class MockChoice:
            def __init__(self, content):
                self.message = type('Message', (), {'content': content})()
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        return MockResponse(result.content)


@dataclass
class ScriptArguments:
    run_name: str = field(
        metadata={"help": "run name to evaluate"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    gpt_ver: str = field(
        default="gpt-4o-mini",
        metadata={"help": "version of GPT to evaluate"},
    )
    every_k: int = field(
        default=1,
        metadata={
            "help": "evaluate every k samples, if a fraction, evaluate each sample 1/k times"
        },
    )
    batch_size: int = field(
        default=256,
        metadata={"help": "batch size for parallel calling chat api"},
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "max retries for calling chat api"},
    )
    max_tokens: int = field(
        default=1000,
        metadata={"help": "max tokens requested for each completion"},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "temperature for sampling"},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "top p for sampling"},
    )
    frequency_penalty: float = field(
        default=0,
        metadata={"help": "frequency penalty for sampling"},
    )
    presence_penalty: float = field(
        default=0,
        metadata={"help": "presence penalty for sampling"},
    )
    is_pairwise: bool = field(
        default=True,
        metadata={"help": "whether the prompt is pairwise comparison"},
    )
    system_prompt: str = field(
        default="none",
        metadata={"help": "system prompt for GPT, do not change"},
    )
    user_prompt: str = field(
        default="none",
        metadata={"help": "user prompt for GPT, do not change"},
    )
    prompt_tokens: int = field(
        default=-1,
        metadata={"help": "estimated tokens per prompt, do not change"},
    )
    model_cache_dir: str = field(
        default=CACHE_CONFIGS["model_cache_dir"],
        metadata={"help": "model cache directory"},
    )
    dataset_cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "dataset cache directory"},
    )


# Load dataset and remove unnecessary columns
def load_generated_dataset(script_args):
    response_dataset = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run_name,
        name=script_args.tag,
        cache_dir=script_args.dataset_cache_dir,
    )

    return response_dataset["default"]


# Call openai api in async manner with rate limit, support criteria and retries
async def call_openai_api(
    request, criteria, max_retries, client, model_name, rate_limiter
):
    request["model"] = model_name
    counter = 0
    global EXTRA_RETRIES
    
    while counter < max_retries:
        async with rate_limiter:
            try:
                result = await asyncio.wait_for(
                    client.chat_completions_create(**request), timeout=60
                )
                # Handle both single result and list of results
                if isinstance(result, list):
                    # Multiple results from list of message lists
                    contents = []
                    for r in result:
                        content = r.choices[0].message.content
                        if isinstance(content, str) and criteria(content):
                            contents.append(content)
                        else:
                            contents.append(None)
                    return contents
                else:
                    # Single result
                    content = result.choices[0].message.content
                    if isinstance(content, str) and criteria(content):
                        return content
            except asyncio.TimeoutError:
                print(f"Request timed out (attempt {counter + 1}/{max_retries})")
                counter += 1
                if counter > 1:
                    EXTRA_RETRIES += 1
                    print(f"Extra retry #{EXTRA_RETRIES} for timeout")
            except Exception as e:
                print(f"Error (attempt {counter + 1}/{max_retries}): {e}")
                counter += 1
                if counter > 1:
                    EXTRA_RETRIES += 1
                    print(f"Extra retry #{EXTRA_RETRIES} for error: {e}")
    return None


async def iterate_call_openai_api(
    request, criteria, max_retries, client, model_name, rate_limiter
):
    # For HUITOpenAI, we can handle all messages in a single call
    # since our wrapper processes the list of message lists
    content = await call_openai_api(
        request=request,
        criteria=criteria,
        max_retries=max_retries,
        client=client,
        model_name=model_name,
        rate_limiter=rate_limiter,
    )
    
    # If content is a list, return it directly
    if isinstance(content, list):
        return content
    else:
        # If it's a single result, wrap it in a list
        return [content]


# Async distribute multiple api calls to multiple clients
async def round_robin_calls(requests, endpoints, criteria, max_retries):
    tasks = [
        iterate_call_openai_api(
            request=request, criteria=criteria, max_retries=max_retries, **endpoint
        )
        for request, endpoint in zip(requests.values(), itertools.cycle(endpoints))
    ]
    contents = await tqdm_asyncio.gather(*tasks)
    return {idx: content for idx, content in zip(requests.keys(), contents)}


# Evaluate reward of responses
async def evaluate_gpt(response_dataset, script_args):
    generator, num_iters = sample_every_k_batched(
        response_dataset, script_args.every_k, batch_size=script_args.batch_size
    )
    endpoints = [
        {
            "client": AsyncHUITOpenAI(
                HUITOpenAI(
                    model=client_config["model_name"],
                    api_key=os.getenv("HUIT_API_KEY"),
                )
            ),
            "model_name": client_config["model_name"],
            "rate_limiter": AsyncLimiter(
                min(
                    client_config["RPM"],
                    client_config["TPM"]
                    // (script_args.prompt_tokens + script_args.max_tokens),
                ),
                60,
            ),
        }
        for client_config in OPENAI_CONFIGS[script_args.gpt_ver]
    ]

    results = {}
    for indices, samples in tqdm(generator, total=num_iters):
        # Populating api requests
        requests = {}
        switch_orders = {}
        print('Indices', len(indices))
        print(indices)
        for idx, sample in zip(indices, samples):
            # Randomly switch the order of the responses with 50% probability
            switch_orders[idx] = np.random.rand() > 0.5
            if script_args.is_pairwise:
                messages = [
                    [
                        {"role": "system", "content": script_args.system_prompt},
                        {
                            "role": "user",
                            "content": script_args.user_prompt.format(
                                prompt=sample["prompt"],
                                # Default to the first response as chosen and the second as the model's response
                                answer1=(
                                    sample["chosen"]
                                    if not switch_orders[idx]
                                    else sample["response"]
                                ),
                                answer2=(
                                    sample["response"]
                                    if not switch_orders[idx]
                                    else sample["chosen"]
                                ),
                            ),
                        },
                    ]
                ]
            else:
                messages = [
                    [
                        {"role": "system", "content": script_args.system_prompt},
                        {
                            "role": "user",
                            "content": script_args.user_prompt.format(
                                prompt=sample["prompt"],
                                answer=(
                                    sample["chosen"]
                                    if not switch_orders[idx]
                                    else sample["response"]
                                ),
                            ),
                        },
                    ],
                    [
                        {"role": "system", "content": script_args.system_prompt},
                        {
                            "role": "user",
                            "content": script_args.user_prompt.format(
                                prompt=sample["prompt"],
                                answer=(
                                    sample["response"]
                                    if not switch_orders[idx]
                                    else sample["chosen"]
                                ),
                            ),
                        },
                    ],
                ]
            requests[idx] = {
                # Leave the model field as None for the chat api to choose the model
                "model": None,
                "messages": messages,
                "max_tokens": script_args.max_tokens,
                "temperature": script_args.temperature,
                "top_p": script_args.top_p,
                "frequency_penalty": script_args.frequency_penalty,
                "presence_penalty": script_args.presence_penalty,
                "stop": None,
            }

        # Call the chat api and ensure the criteria
        contents = await round_robin_calls(
            requests=requests,
            endpoints=endpoints,
            criteria=lambda content: any(
                bool(re.search(pattern, content))
                for pattern in script_args.match_patterns
            ),
            max_retries=script_args.max_retries,
        )

        # Collecting results
        for idx, content in contents.items():
            # Skip if the content is None
            if content[0] is None or (
                not script_args.is_pairwise and content[1] is None
            ):
                continue
            if idx not in results:
                results[idx] = []
            # Parse the scores and explanation from the chat api response
            # Because of ensured criteria, this will always succeed
            for pattern in script_args.match_patterns:
                if re.search(pattern, content[0]):
                    if script_args.is_pairwise:
                        score1, score2 = tuple(
                            map(float, re.findall(pattern, content[0])[-1])
                        )
                    else:
                        score1 = float(re.findall(pattern, content[0])[-1])
                        score2 = float(re.findall(pattern, content[1])[-1])
                        break
            # Reverse the scores if the order was switched
            results[idx].append(
                (score2 - score1) if not switch_orders[idx] else (score1 - score2)
            )
    print('Results: ')
    print(results)
    # Add or replace the 'gpt_score' column in the dataset
    response_dataset = (
        response_dataset.remove_columns("gpt_score")
        if "gpt_score" in response_dataset.column_names
        else response_dataset
    )
    response_dataset = response_dataset.add_column(
        "gpt_score",
        [
            np.mean(results[idx]) if idx in results else None
            for idx in range(len(response_dataset))
        ],
    )
    return response_dataset


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Dataset
    print('Loading datasets')
    response_dataset = load_generated_dataset(script_args)
    response_dataset = response_dataset.to_pandas()
    # print((response_dataset['gpt_score'].dropna()>0).mean())
    # raise
    response_dataset['prompt'] = response_dataset['prompt'].apply(lambda x: x.replace( "Format your code with Python code block: ```python\n[YOUR FUNCTION]\n```", "Format your code with triple $ signs: $$$[YOUR FUNCTION]$$$"))
    response_dataset['prompt'] = response_dataset['prompt'].apply(lambda x: x.replace("\n```python\n", "'$$$ "))
    response_dataset['prompt'] = response_dataset['prompt'].apply(lambda x: x.replace("$$$'", "\n```"))
    from datasets import Dataset
    response_dataset = Dataset.from_pandas(response_dataset)
    print('Response Data size: ', len(response_dataset))

    # Find the GPT template by dataset type
    for dataset_prefix in GPT_CONFIGS.keys():
        if script_args.run_name.split("_")[2].startswith(dataset_prefix):
            script_args.is_pairwise = GPT_CONFIGS[dataset_prefix]["is_pairwise"]
            script_args.system_prompt = GPT_CONFIGS[dataset_prefix]["system_prompt"]
            script_args.user_prompt = GPT_CONFIGS[dataset_prefix]["user_prompt"]
            script_args.match_patterns = GPT_CONFIGS[dataset_prefix]["match_patterns"]
            script_args.prompt_tokens = (
                len(
                    TOKENIZER.encode(
                        script_args.system_prompt
                        + script_args.user_prompt
                        + response_dataset[0]["prompt"]
                        + response_dataset[0]["chosen"]
                        + response_dataset[0]["response"]
                    )
                )
                + 100
            )
            break

    # Evaluation
    response_dataset = asyncio.run(evaluate_gpt(response_dataset, script_args))

    # Print cost information
    print(f"Total cost for this evaluation: ${COST:.4f}")
    print(f"Total extra retries required: {EXTRA_RETRIES}")

    # Push to Hub
    DatasetDict(
        {"default": response_dataset},
    ).push_to_hub(
        HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run_name+'g',
        script_args.tag,
    )


if __name__ == "__main__":
    main()
