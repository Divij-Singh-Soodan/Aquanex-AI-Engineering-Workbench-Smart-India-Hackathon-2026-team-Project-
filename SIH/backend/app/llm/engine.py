"""
Aquanex - Unified LLM Serving Engine (app/llm/engine.py)
Orchestrates enterprise foundation models for air-gapped refinery engineering:
- Laptop/Edge (8B/3B Tier): Ollama OpenAI-compatible endpoint (http://127.0.0.1:11434/v1)
- Datacenter (100B+ Tier): vLLM / SGLang multi-GPU cluster (e.g., http://127.0.0.1:8001/v1)

Hardware Scaling Formula:
- Laptop:   Qwen 2.5 3B/8B, DeepSeek-R1 8B via Ollama
- Cluster:  DeepSeek-R1 671B/70B, Qwen-2.5 72B, Qwen2-VL 72B via vLLM TP=4

Strict Air-Gapped Policy:
- If no model server is connected, the system refuses to run and raises ConnectionError.
- Emulation is strictly disabled unless explicitly opted-in via settings.ALLOW_EMULATION = True.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
import httpx

from app.config import EDGE_LAPTOP_CONTEXT_WINDOW, settings

logger = logging.getLogger("sovereign_llm_engine")


def normalize_v1_url(url: str) -> str:
    """Ensure URL has scheme and ends with /v1 without duplicate slashes."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    url = re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url, flags=re.IGNORECASE)
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def ollama_native_root(v1_url: str) -> str:
    """Strip a trailing /v1 so native /api/chat and /api/generate hit the daemon."""
    url = v1_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


# Applied on every Ollama body. num_gpu=99 offloads every layer; num_ctx stays
# inside the RTX 3050 6GB budget so Ollama does not spill the model onto CPU.
OLLAMA_GPU_OPTIONS = {
    "num_gpu": 99,
    "num_ctx": EDGE_LAPTOP_CONTEXT_WINDOW,
}


def ollama_request_options(temperature: float = 0.2, max_tokens: int = 4096) -> Dict[str, Any]:
    """Options block required on /api/chat, /api/generate, and /v1/chat/completions."""
    return {
        "num_gpu": OLLAMA_GPU_OPTIONS["num_gpu"],
        "num_ctx": OLLAMA_GPU_OPTIONS["num_ctx"],
        "temperature": temperature,
        "num_predict": max_tokens,
    }


class ModelInferenceError(RuntimeError):
    """Failure details shared by the synchronous and streaming chat endpoints."""

    def __init__(self, message: str, code: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ModelServerOfflineError(ModelInferenceError, ConnectionError):
    """All candidate model servers failed to accept a connection."""

    def __init__(self, endpoints: str):
        super().__init__(
            f"NO LLM MODEL CONNECTED: Local model server is offline. Could not connect to {endpoints}.",
            "FAILED_NO_MODEL",
            503,
        )


class FoundationModelEngine:
    """
    Production-grade enterprise LLM gateway with native vLLM and Ollama integration.
    Strictly fails if no model server is running.
    """

    def __init__(self):
        self.vllm_url = normalize_v1_url(getattr(settings, "VLLM_HOST", "http://127.0.0.1:8001/v1"))
        self.ollama_url = normalize_v1_url(getattr(settings, "OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.base_url = self.vllm_url
        self.timeout = httpx.Timeout(120.0, connect=1.5)
        self.ollama_timeout = httpx.Timeout(300.0, connect=5.0)

    async def check_cluster_health(self) -> Dict[str, Any]:
        """
        Probe connected model servers (vLLM datacenter or Ollama workstation/laptop)
        to determine live GPU topology, active weights, and connection status.
        """
        profile = getattr(settings, "active_model_profile", {})
        tier = getattr(settings, "MODEL_TIER", "ENTERPRISE_100B")

        telemetry: Dict[str, Any] = {
            "tier": tier,
            "profile_name": profile.get("tier_name", "Enterprise Refinery Datacenter (100B+)"),
            "target_hardware": profile.get("hardware_spec", "Dual AMD EPYC 9654 + 4x NVIDIA A100 (80GB SXM4)"),
            "tensor_parallel_size": getattr(settings, "TENSOR_PARALLEL_SIZE", 4),
            "max_context_window": getattr(settings, "MAX_MODEL_LEN", 131072),
            "primary_model": profile.get("reasoning", getattr(settings, "MODEL_REASONING", "deepseek-r1:70b-q8_0")),
            "vision_model": profile.get("vision", getattr(settings, "MODEL_VISION", "qwen2-vl:72b-instruct")),
            "coder_model": profile.get("coder", getattr(settings, "MODEL_CODER", "qwen2.5-coder:32b-instruct")),
            "router_model": profile.get("router", getattr(settings, "MODEL_ROUTER", "qwen2.5:72b-instruct")),
            "vllm_endpoint": self.vllm_url,
            "ollama_endpoint": self.ollama_url,
            "is_connected_to_vllm": False,
            "is_connected_to_ollama": False,
            "is_connected": False,
            "active_endpoint": None,
            "mode": "DISCONNECTED",
            "datacenter_node_status": "OFFLINE - No model server reachable (vLLM or Ollama)",
            "available_models": [],
        }

        # 1. Probe vLLM Cluster (Datacenter)
        try:
            async with httpx.AsyncClient(timeout=1.5, trust_env=False) as client:
                res = await client.get(f"{self.vllm_url}/models")
                if res.status_code == 200:
                    models_data = res.json().get("data", [])
                    telemetry["is_connected_to_vllm"] = True
                    telemetry["is_connected"] = True
                    telemetry["active_endpoint"] = self.vllm_url
                    telemetry["mode"] = "CONNECTED_DATACENTER_VLLM"
                    telemetry["datacenter_node_status"] = "LIVE_GPU_CLUSTER_ACTIVE"
                    telemetry["available_models"] = [m.get("id") for m in models_data]
                    return telemetry
        except Exception as e:
            logger.debug(f"vLLM cluster offline ({self.vllm_url}): {e}")

        # 2. Probe Ollama Node (Workstation / Laptop)
        try:
            async with httpx.AsyncClient(timeout=1.5, trust_env=False) as client:
                res = await client.get(f"{self.ollama_url}/models")
                if res.status_code == 200:
                    models_data = res.json().get("data", [])
                    telemetry["is_connected_to_ollama"] = True
                    telemetry["is_connected"] = True
                    telemetry["active_endpoint"] = self.ollama_url
                    telemetry["mode"] = "CONNECTED_LOCAL_OLLAMA"
                    telemetry["datacenter_node_status"] = "LIVE_OLLAMA_NODE_ACTIVE"
                    telemetry["available_models"] = [m.get("id") for m in models_data]
                    return telemetry
        except Exception as e:
            logger.debug(f"Ollama server offline ({self.ollama_url}): {e}")

        return telemetry

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str = "",
        model_type: str = "reasoning",  # "reasoning", "vision", "coder", "general", "router"
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate a response using live connected models (vLLM or Ollama).
        Returns: (content, thought_trace)

        Raises ModelInferenceError with the upstream failure details.
        """
        profile = getattr(settings, "active_model_profile", {})
        if model_type == "vision":
            model_name = profile.get("vision", getattr(settings, "MODEL_VISION", "qwen2-vl:72b-instruct"))
        elif model_type == "coder":
            model_name = profile.get("coder", getattr(settings, "MODEL_CODER", "qwen2.5-coder:32b-instruct"))
        elif model_type == "router":
            model_name = profile.get("router", getattr(settings, "MODEL_ROUTER", "qwen2.5:72b-instruct"))
        else:
            model_name = profile.get("reasoning", getattr(settings, "MODEL_REASONING", "deepseek-r1:70b-q8_0"))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try live model endpoints in priority order:
        # Datacenter vLLM first, then Workstation/Laptop Ollama.
        # Ollama is dispatched on native /api/chat so options.num_gpu is honored.
        # The OpenAI shim at /v1/chat/completions drops that field on several releases.
        candidate_endpoints = [
            ("vLLM (Datacenter)", self.vllm_url, "openai"),
            ("Ollama (Local/Workstation)", self.ollama_url, "ollama"),
        ]

        failure = None
        for server_label, base_url, api_kind in candidate_endpoints:
            is_ollama = api_kind == "ollama" or "11434" in base_url
            timeout = self.ollama_timeout if is_ollama else self.timeout
            request = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                    res = await self._post_chat(client, base_url, api_kind, request)
                    if res.status_code == 404:
                        try:
                            fallback_model = await self._find_fallback_model(
                                client, base_url, is_ollama, model_type, model_name
                            )
                            if fallback_model:
                                logger.info("Auto-mapping requested %r to installed model %r", model_name, fallback_model)
                                request["model"] = fallback_model
                                res = await self._post_chat(client, base_url, api_kind, request)
                        except (httpx.ConnectError, httpx.ConnectTimeout):
                            # Keep the HTTP error from the original request if recovery cannot connect.
                            request["model"] = model_name
                            res.raise_for_status()
                    res.raise_for_status()
                    raw_content = self._content_from_chat_response(res)
                    if raw_content is not None:
                        thought, content = self._extract_deepseek_r1_thought(raw_content)
                        return content, thought
                    failure = ModelInferenceError(
                        f"{server_label} returned no assistant content: {res.text}",
                        "MODEL_RESPONSE_ERROR",
                        502,
                    )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                logger.debug("%s connection failed at %s: %s", server_label, base_url, exc)
                continue
            except httpx.ReadTimeout as exc:
                failure = ModelInferenceError(
                    f"{server_label} did not respond within the {timeout.read:g}-second read timeout "
                    f"while requesting {exc.request.url} for model {request['model']!r}.",
                    "MODEL_READ_TIMEOUT",
                    504,
                )
            except httpx.HTTPStatusError as exc:
                response = exc.response
                detail = response.text.strip() or response.reason_phrase
                failure = ModelInferenceError(
                    f"{server_label} returned HTTP {response.status_code} for "
                    f"{response.request.url} (model {request['model']!r}): {detail}",
                    "MODEL_HTTP_ERROR",
                    502,
                )
            except (httpx.RequestError, ValueError) as exc:
                failure = ModelInferenceError(
                    f"{server_label} request for model {request['model']!r} failed: {exc}",
                    "MODEL_RESPONSE_ERROR",
                    502,
                )
            logger.warning("%s", failure)

        if failure is not None:
            raise failure

        if getattr(settings, "ALLOW_EMULATION", False):
            logger.info("ALLOW_EMULATION=True: using synthetic execution fallback.")
            return self._emulate_100b_execution(prompt, system_prompt, model_type)

        raise ModelServerOfflineError(f"{self.ollama_url} (Ollama) or {self.vllm_url} (vLLM)")

    async def _find_fallback_model(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        is_ollama: bool,
        model_type: str,
        requested_model: str,
    ) -> Optional[str]:
        """Choose an installed model after the server rejects the requested tag."""
        if is_ollama:
            response = await client.get(f"{ollama_native_root(base_url)}/api/tags")
            response.raise_for_status()
            models = [model.get("name") or model.get("model") for model in response.json().get("models", [])]
        else:
            response = await client.get(f"{base_url}/models")
            response.raise_for_status()
            models = [model.get("id") for model in response.json().get("data", [])]

        available = [model for model in models if model]
        if not available or requested_model in available:
            return None
        if is_ollama:
            return available[0]
        wants_coder = model_type == "coder"
        return next((model for model in available if ("coder" in model.lower()) == wants_coder), available[0])

    async def _post_chat(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_kind: str,
        request: Dict[str, Any],
    ) -> httpx.Response:
        """
        Post one chat turn.

        Ollama always receives native ``/api/chat`` with ``options.num_gpu=99``
        and ``options.num_ctx=4096``. The same options are attached if a caller
        uses the OpenAI-compatible route.
        """
        if api_kind == "ollama" or "11434" in base_url:
            root = ollama_native_root(base_url)
            options = ollama_request_options(
                temperature=request.get("temperature", 0.2),
                max_tokens=request.get("max_tokens", 4096),
            )
            if "options" in request and isinstance(request["options"], dict):
                options.update(request["options"])
            # Strictly enforce GPU acceleration and context cap for RTX 3050
            options["num_gpu"] = 99
            options["num_ctx"] = 4096

            native_body = {
                "model": request["model"],
                "messages": request["messages"],
                "stream": False,
                "keep_alive": "30m",
                "options": options,
            }
            res = await client.post(f"{root}/api/chat", json=native_body)
            # Older daemons that only expose the OpenAI route still get the GPU options.
            if res.status_code == 404 and "model" not in (res.text or "").lower():
                compat_body = {
                    "model": request["model"],
                    "messages": request["messages"],
                    "temperature": request.get("temperature", 0.2),
                    "max_tokens": request.get("max_tokens", 4096),
                    "options": {
                        "num_gpu": OLLAMA_GPU_OPTIONS["num_gpu"],
                        "num_ctx": OLLAMA_GPU_OPTIONS["num_ctx"],
                    },
                }
                try:
                    return await client.post(f"{base_url}/chat/completions", json=compat_body)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    res.raise_for_status()
            return res

        return await client.post(f"{base_url}/chat/completions", json=request)

    async def _post_generate(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        request: Dict[str, Any],
    ) -> httpx.Response:
        """
        Post one prompt turn to Ollama's native /api/generate endpoint.
        Strictly includes options: {"num_gpu": 99, "num_ctx": 4096}.
        """
        root = ollama_native_root(base_url)
        options = ollama_request_options(
            temperature=request.get("temperature", 0.2),
            max_tokens=request.get("max_tokens", 4096),
        )
        if "options" in request and isinstance(request["options"], dict):
            options.update(request["options"])
        options["num_gpu"] = 99
        options["num_ctx"] = 4096

        body = {
            "model": request["model"],
            "prompt": request.get("prompt", ""),
            "stream": request.get("stream", False),
            "keep_alive": "30m",
            "options": options,
        }
        if "system" in request:
            body["system"] = request["system"]
        return await client.post(f"{root}/api/generate", json=body)

    def _content_from_chat_response(self, res: Optional[httpx.Response]) -> Optional[str]:
        """Read assistant text from an OpenAI chat body or a native Ollama body."""
        if res is None or res.status_code != 200:
            return None
        data = res.json()
        choices = data.get("choices") or []
        if choices:
            return choices[0].get("message", {}).get("content", "")
        message = data.get("message")
        if isinstance(message, dict) and "content" in message:
            return message.get("content") or ""
        if isinstance(data.get("response"), str):
            return data["response"]
        return None

    def _extract_deepseek_r1_thought(self, text: str) -> Tuple[Optional[str], str]:
        """Extract DeepSeek-R1 <think>...</think> reasoning traces."""
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return thought, content
        return None, text

    def _emulate_100b_execution(
        self,
        prompt: str,
        system_prompt: str,
        model_type: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Emulate pure 100B execution contract with high-fidelity DeepSeek-R1 CoT
        and deterministic structured outputs (Only used when ALLOW_EMULATION=True in tests).
        """
        p_upper = prompt.upper()
        s_upper = system_prompt.upper()

        if model_type == "vision" or "VISION" in s_upper or "P&ID" in p_upper or "ULTRASONIC" in p_upper:
            thought = (
                "DeepSeek-R1 (671B MoE) / Qwen2-VL-72B Visual Analysis:\n"
                "1. Analyzing uploaded refinery asset (P&ID Sheet 4 / Ultrasonic NDT Thickness Log).\n"
                "2. Identifying line tag 'CDU-2-04-150-A1A' in Crude Distillation Unit 2 column overhead.\n"
                "3. Reading ultrasonic grid thickness measurements: nominal t_nom = 4.8 mm, measured t_act = 3.2 mm.\n"
                "4. ASME B31.3 Schedule 40 Carbon Steel design limits: P = 150 psi, T = 135 deg C.\n"
                "5. Determining retirement minimum thickness t_min = 2.1 mm per Section 7.1.1 API 570."
            )
            content = json.dumps({
                "line_tag": "CDU-2-04-150-A1A",
                "material": "ASTM A106 Grade B Carbon Steel",
                "nominal_thickness_mm": 4.8,
                "actual_thickness_mm": 3.2,
                "design_pressure_psi": 150.0,
                "design_temp_celsius": 135.0,
                "unit": "Crude Distillation Unit 2 (CDU-2)",
                "fluid_service": "Hydrocarbon Vapor / Sour Gas",
                "corrosion_mechanism": "Sulfidation & Wet H2S Internal Thinning",
            }, indent=2)
            return content, thought

        elif model_type == "coder" or "CODE" in s_upper or "SANDBOX" in s_upper:
            thought = (
                "Code Execution Preparation:\n"
                "Synthesizing deterministic Python verification script for API 570 remaining life calculation."
            )
            content = (
                "def calculate_api570(t_nom=4.8, t_act=3.2, t_min=2.1, years=10.0):\n"
                "    corrosion_rate = (t_nom - t_act) / years\n"
                "    remaining_life = (t_act - t_min) / corrosion_rate if corrosion_rate > 0 else 99.0\n"
                "    return {'corrosion_rate': corrosion_rate, 'remaining_life_years': remaining_life}\n"
            )
            return content, thought

        elif "ROUTER" in s_upper:
            thought = "Task Router: Classifying industrial refinery query intent."
            return "VISION_AUDIT", thought

        # General industrial query
        thought = (
            "Engineering Query Analysis:\n"
            "Retrieving MRPL Refinery Operations & Engineering Manual standard operating limits."
        )
        content = (
            "Verified against MRPL refinery operating manuals and statutory ASME B31.3 / API 570 guidelines. "
            "All calculations executed deterministically within isolated sandbox environments."
        )
        return content, thought


# Global singleton instance
foundation_engine = FoundationModelEngine()


def get_llm_engine() -> FoundationModelEngine:
    """Returns the singleton FoundationModelEngine gateway instance."""
    return foundation_engine
