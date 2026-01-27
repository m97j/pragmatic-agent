# app/models/initializer.py
import textwrap
from typing import TypedDict, Union

import onnxruntime as ort
import torch
import torch.serialization
from huggingface_hub import hf_hub_download
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer)

import config
from models.architectures.qwen_extension import CustomModel


class ModelDict(TypedDict):
    llm: Union[CustomModel, torch.nn.Module]
    llm_tokenizer: PreTrainedTokenizer
    reranker: ort.InferenceSession
    reranker_tokenizer: PreTrainedTokenizer

_MODELS: dict[str, ModelDict] = {}
_PREFIX_CACHE = {}

def download_llm() -> tuple[str, str]:
    """
    Download the quantized LLM file from Hugging Face Hub 
    (e.g., model_quantized.pt or model.bin).
    Returns the local path to the model and config files.
    """
    local_model_path = hf_hub_download(
        repo_id=config.HF_MODEL_HUB,
        filename=config.HF_LLM_FILENAME,
        token=config.HF_TOKEN
    )

    local_config_path = hf_hub_download(
        repo_id=config.HF_MODEL_HUB,
        filename=config.HF_CONFIG_FILENAME,
        token=config.HF_TOKEN
    )
    return local_model_path, local_config_path

def download_reranker() -> str:
    """
    Download the reranker ONNX file from Hugging Face Hub.
    Returns the local path to the reranker file.
    """
    return hf_hub_download(
        repo_id=config.HF_MODEL_HUB,
        filename=config.HF_RERANKER_FILENAME,
        token=config.HF_TOKEN
    )

def load_llm(local_model_path: str, local_config_path: str) -> CustomModel:
    """
    Load the quantized LLM into PyTorch.
    If the model file is named 'pytorch_model.bin', from_pretrained will load it automatically.
    Otherwise, fall back to manual state_dict loading.
    """
    torch.serialization.add_safe_globals([AffineQuantizedTensor])

    _config = AutoConfig.from_pretrained(local_config_path)
    model = CustomModel(_config)
    state_dict = torch.load(local_model_path, map_location=torch.device, weights_only=True)
    model.load_state_dict(state_dict)

    return model

def load_reranker(local_model_path: str) -> ort.InferenceSession:
    """
    Load reranker model with ONNX Runtime.
    """
    return ort.InferenceSession(local_model_path, providers=["CPUExecutionProvider"])

def load_llm_tokenizer() -> PreTrainedTokenizer:
    """
    Load tokenizer for LLM
    """
    return AutoTokenizer.from_pretrained(
        config.HF_LLM_REPO,
        token=config.HF_TOKEN
    )

def load_reranker_tokenizer() -> PreTrainedTokenizer:
    """
    Load tokenizer for reranker
    """
    return AutoTokenizer.from_pretrained(
        config.HF_RERANKER_REPO,
        token=config.HF_TOKEN
    )

def load_llm_from_pretrained() -> CustomModel:
    """
    Load the official LLM (e.g., 4B model) directly from Hugging Face Hub
    using from_pretrained. This bypasses local quantized state_dict loading.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.HF_LLM_REPO,
        token=config.HF_TOKEN,
        dtype=torch.float16,   
        device_map=torch.device  
    )
    return model

def initialize_models() -> None:
    """
    Download and load models on first run, then save to global cache.
    """
    global _MODELS
    if not _MODELS:
        # llm_path, config_path = download_llm()
        reranker_path = download_reranker()

        # _MODELS["llm"] = load_llm(llm_path, config_path)
        _MODELS["llm"] = load_llm_from_pretrained()
        _MODELS["llm_tokenizer"] = load_llm_tokenizer()

        _MODELS["reranker"] = load_reranker(reranker_path)
        _MODELS["reranker_tokenizer"] = load_reranker_tokenizer()

def get_models() -> ModelDict:
    """
    Retrieve models and tokenizers from cache.
    """
    global _MODELS
    if not _MODELS:
        initialize_models()
    return _MODELS

def initialize_prefixes() -> dict[str, torch.Tensor]:
    """
    Initialize prefix cache once and store globally.
    Each entry is stored as a torch.Tensor of input_ids.
    """
    global _PREFIX_CACHE
    if not _PREFIX_CACHE:
        models = get_models()
        tokenizer = models["llm_tokenizer"]
        _PREFIX_CACHE = {
            "instruct": tokenizer("/no_think\n", return_tensors="pt")["input_ids"],
            "think": tokenizer("/think\n", return_tensors="pt")["input_ids"],
            "summarize": tokenizer(textwrap.dedent("""\
                Instruction: Summarize the following document in relation to the query
                Constraints:
                - Keep the summary under 300 words
                - Focus only on information relevant to the query
                - Maintain the original language of the document
            """), return_tensors="pt")["input_ids"],
            "refine": tokenizer(textwrap.dedent("""\
                Instruction: Combine and refine these summaries to answer the query
                Constraints:
                - Provide the final answer in a single coherent paragraph
                - Ensure the answer directly addresses the query
                - Keep the length under 500 words
                - Preserve the language style of the input summaries
            """), return_tensors="pt")["input_ids"],
            "query": tokenizer("query: \n", return_tensors="pt")["input_ids"],
            "document": tokenizer("document: \n", return_tensors="pt")["input_ids"],
            "summaries": tokenizer("summaries:\n", return_tensors="pt")["input_ids"],
            "summarize_reminder": tokenizer("Reminder: Keep the summary concise and under 300 words.", 
                                            return_tensors="pt")["input_ids"],
            "refine_reminder": tokenizer("Reminder: Final answer must be a single coherent paragraph under 500 words.", 
                                         return_tensors="pt")["input_ids"],
            "newline": tokenizer("\n", return_tensors="pt")["input_ids"],
        }
    return _PREFIX_CACHE

def get_prefixes() -> dict[str, torch.Tensor]:
    """
    Retrieve prefix cache from global storage.
    """
    global _PREFIX_CACHE
    if not _PREFIX_CACHE:
        initialize_prefixes()
    return _PREFIX_CACHE
