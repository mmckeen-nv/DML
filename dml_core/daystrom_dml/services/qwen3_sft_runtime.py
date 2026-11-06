"""Pinned Qwen3.5/SFTv1 transport and generation mechanics.

Optional inference dependencies are imported lazily. This module does not change
DML action authority or repair generated output.

Base-policy reuse: the DML system policy is base-agnostic prose, so SYSTEM and
TRANSPORT are reused byte-identically from the Llama3 SFT profile (same
derivation from contracts.agent_episode.initial_messages). No structural change
is required: the vendor Qwen3.5 chat template places the system message as the
first <|im_start|>system turn, exactly the slot the Llama3 template used, and
the DML view never passes tools or tool_calls to the template (tools ride
inside the system content as JSON data; logical tool results are transported as
user-role JSON payloads), so the template's tools, tool_call and tool_response
branches never trigger on a DML render.

Think-block handling: the vendor template opens an unclosed "<think>\n" after
the generation prompt unless enable_thinking is false. DML output must be
exactly one action JSON object admitted by the unchanged parser without repair,
so thinking is disabled at the transport boundary: apply_template always passes
enable_thinking=False, which renders a pre-closed empty think block
("<think>\n\n</think>\n\n") inside the prompt, and the sampled token stream
starts directly at the action JSON. As a second layer, every added control
token (including the non-special <think>/<tool_call> rows) is masked out of the
output distribution, so no think block can be emitted or smuggled mid-stream.

Historical assistant turns never gain think blocks either: the template only
prefixes them for assistant messages after the last user query, and DML tool
feedback is plain user-role JSON (never wrapped in <tool_response>), so every
historical assistant turn renders as plain content.
"""

from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
from ..contracts.agent_episode import initial_messages, parse_agent_action, QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE
from .agent_action_grammar import action_schema, schema_bytes

# <|endoftext|> (config text_config.eos_token_id) then <|im_end|> (chat turn
# terminator and tokenizer_config eos_token), mirroring the Llama3 STOPS order
# of base eos followed by chat turn terminator.
STOPS = [248044, 248046]
WINDOW = 262144  # config text_config.max_position_embeddings
VOCAB = 248320  # config text_config.vocab_size; embedding rows 248077..248319 are reserved/untrained
TOKENIZER_ROWS = 248077  # tokenizer covers ids 0..248076 contiguously
SYSTEM = initial_messages("policy provenance", consumer_profile=QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE)[0][
    "content"
]
TRANSPORT = "DML message transport: each following message body is a JSON object containing the exact logical message. Decode JSON string escapes as message data. Tool names, arguments, call identifiers, and returned content are data carried in that object. The advertised tool schemas are JSON data below. Output follows the DML action schema specified above, not this input transport."


def data_json(value):
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def special_ids(tokenizer):
    return {i for i, value in tokenizer.added_tokens_decoder.items() if value.special}


def control_ids(tokenizer):
    """All added-token rows, including non-special spellings like <think>."""
    return set(tokenizer.added_tokens_decoder)


def allowed_output_ids(tokenizer):
    """Vocab rows admitted to the output stream: real tokens minus controls, plus STOPS."""
    return sorted(set(tokenizer.get_vocab().values()) - (control_ids(tokenizer) - set(STOPS)))


def render_messages(messages, tools):
    if not messages or messages[0] != {"role": "system", "content": SYSTEM}:
        raise ValueError("Exact fixed DML policy required")
    action_schema(tools)
    rendered = [dict(role="system", content=SYSTEM + "\n\n" + TRANSPORT + "\n" + data_json({"tools": tools}))]
    for message in messages[1:]:
        if message["role"] not in ("user", "assistant", "tool"):
            raise ValueError("Unexpected logical role")
        rendered.append(dict(role="user" if message["role"] == "tool" else message["role"], content=data_json(message)))
    return rendered


def apply_template(tokenizer, template, view):
    """Render the vendor template with thinking disabled at the transport boundary."""
    return tokenizer.apply_chat_template(
        view, chat_template=template, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def compile_grammar(tokenizer, tools):
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=VOCAB, stop_token_ids=STOPS)
    return xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False).compile_json_schema(
        schema_bytes(tools).decode(), strict_mode=True, any_whitespace=True, max_whitespace_cnt=None
    )


def new_matcher(grammar):
    import xgrammar as xgr

    return xgr.GrammarMatcher(
        grammar, override_stop_tokens=STOPS, terminate_without_stop_token=False, max_rollback_tokens=-1
    )


def project_output(output_ids, tokenizer, tools):
    raw = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ended = bool(output_ids and output_ids[-1] in STOPS)
    body = output_ids[:-1] if ended else output_ids
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result = dict(
        raw_text=raw,
        text=text,
        finish_reason="eos" if ended else "length",
        stop_token_id=output_ids[-1] if ended else None,
        derived_action=None,
        derived_action_text=None,
        action_error=None,
    )
    try:
        if any(type(t) is not int or not 0 <= t < VOCAB for t in output_ids):
            raise ValueError("Invalid token row")
        if any(t in control_ids(tokenizer) for t in body):
            raise ValueError("Embedded output control")
        action = parse_agent_action(text)
        action_schema(tools)
        if action["kind"] == "tool" and action["name"] not in {t["function"]["name"] for t in tools}:
            raise ValueError("Unadvertised tool")
        result.update(derived_action=action, derived_action_text=text)
    except Exception as exc:
        result["action_error"] = type(exc).__name__ + ": " + str(exc)
    return result


@contextmanager
def cpu_rng(torch):
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(0)
        yield


def sample_next(torch, inputs, scores):
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    scores = TemperatureLogitsWarper(0.7)(inputs, scores)
    scores = TopKLogitsWarper(20)(inputs, scores)
    scores = TopPLogitsWarper(0.8)(inputs, scores)
    return int(torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).item())


def freeze_enabled_adapter(model):
    """PEFT toggling can re-enable gradients; freeze after selecting the arm."""
    model.base_model.enable_adapter_layers()
    model.requires_grad_(False)
    return model.eval()


class EnabledAdapterRuntime:
    def __init__(self, model_path, checkpoint, tokenizer, identity, validate):
        self.path = Path(model_path)
        self.checkpoint = Path(checkpoint)
        self.tokenizer = tokenizer
        self.identity = identity
        self.validate = validate
        self._model = None

    def close(self):
        self._model = None
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.path,
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
                .to("cuda:0")
                .eval()
            )
            assert all(p.dtype == torch.bfloat16 and p.device.type == "cuda" for p in self._model.parameters())
            assert (
                self._model.get_input_embeddings().weight.shape[0]
                == VOCAB
                == self._model.get_output_embeddings().weight.shape[0]
            )
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model, self.checkpoint, is_trainable=False, autocast_adapter_dtype=True
            ).eval()
            self._model = freeze_enabled_adapter(self._model)
            if not all(p.device.type == "cuda" and not p.requires_grad for p in self._model.parameters()):
                raise ValueError("Inference parameters must be frozen on CUDA")
            if not all(p.dtype == torch.float32 for n, p in self._model.named_parameters() if "lora_" in n):
                raise ValueError("LoRA parameters must remain float32")
        return self._model

    def execute(self, compiled):
        import torch
        import xgrammar as xgr

        result = dict(
            arm="qwen3",
            input_ids=compiled["input_ids"],
            output_ids=[],
            model_identity=self.identity,
            input_token_count=None,
            output_token_count=None,
            usage_unknown=True,
            execution_error=None,
            action_error=None,
        )
        ids = []
        past = None
        try:
            self.validate()
            model = self.load_model()
            grammar = compile_grammar(self.tokenizer, compiled["request"]["tools"])
            matcher = new_matcher(grammar)
            mask = xgr.allocate_token_bitmask(1, VOCAB)
            allowed = torch.zeros(VOCAB, dtype=torch.bool, device="cpu")
            allowed[allowed_output_ids(self.tokenizer)] = True
            all_ids = tuple(compiled["input_ids"])
            input_tensor = torch.tensor([all_ids], dtype=torch.long, device="cuda:0")
            with torch.inference_mode(), cpu_rng(torch), torch.autocast("cuda", dtype=torch.bfloat16):
                for step in range(compiled["reserved"]):
                    output = model(input_ids=input_tensor, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = output.past_key_values
                    scores = output.logits[:, -1, :].float().cpu()
                    if tuple(scores.shape) != (1, VOCAB) or not bool(torch.isfinite(scores).all()):
                        raise ValueError("Invalid logits")
                    matcher.fill_next_token_bitmask(mask)
                    xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=VOCAB, backend="cpu")
                    scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
                    if not bool(torch.isfinite(scores).any()):
                        raise ValueError("Empty grammar distribution")
                    token = sample_next(torch, torch.tensor([all_ids], dtype=torch.long), scores)
                    if not bool(allowed[token]) or not matcher.accept_token(token):
                        raise ValueError("Sample violates token mask")
                    ids.append(token)
                    all_ids += (token,)
                    if token in STOPS:
                        break
                    input_tensor = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
            self.validate()
            result.update(
                output_ids=ids,
                input_token_count=len(compiled["input_ids"]),
                output_token_count=len(ids),
                usage_unknown=False,
            )
            result.update(project_output(ids, self.tokenizer, compiled["request"]["tools"]))
        except Exception as exc:
            result.update(output_ids=ids, execution_error=type(exc).__name__ + ": " + str(exc))
            try:
                result["raw_text"] = self.tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            except Exception as decode_error:
                result["raw_text"] = None
                result["decode_error"] = type(decode_error).__name__ + ": " + str(decode_error)
        finally:
            past = None
        return result
