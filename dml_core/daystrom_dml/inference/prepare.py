"""DIP preparation boundary around the legacy frontier pipeline."""
from __future__ import annotations

from typing import Any, Optional

from daystrom_dml import utils
from daystrom_dml.api_contracts import ContractError, TokenBudget
from daystrom_dml.cognition.schema import CognitivePacket
from daystrom_dml.frontier_pipeline import DraftGenerator, FrontierCompressionPipeline, FrontierPipelineConfig
from daystrom_dml.inference.schema import DIPPrepareRequest, DIPPrepareResult


class InferencePreparationPipeline:
    """Prepare frontier input without performing frontier inference.

    This is the DIP prototype boundary.  It can wrap an existing
    FrontierCompressionPipeline for backwards-compatible DML context preparation,
    or prepare directly from a DCN cognitive packet when one is supplied.
    """

    def __init__(
        self,
        adapter: Any = None,
        *,
        config: Optional[FrontierPipelineConfig] = None,
        draft_generator: Optional[DraftGenerator] = None,
        context_controller: Any = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or FrontierPipelineConfig()
        self.draft_generator = draft_generator
        self.context_controller = context_controller

    def prepare(self, request: DIPPrepareRequest | dict[str, Any]) -> DIPPrepareResult:
        req = request if isinstance(request, DIPPrepareRequest) else DIPPrepareRequest.from_dict(request)
        if req.cognitive_packet is not None:
            result = self._prepare_from_packet(req, req.cognitive_packet)
        else:
            result = self._prepare_from_prompt(req)
        return self._attach_context_observation(req, result)

    def _prepare_from_packet(self, req: DIPPrepareRequest, packet: CognitivePacket) -> DIPPrepareResult:
        prompt = req.prompt or packet.assembled_context or self._prompt_from_packet(packet)
        frontier_prompt = self._frontier_prompt_from_packet(packet=packet, prompt=prompt)
        frontier_tokens = utils.estimate_tokens(frontier_prompt)
        return DIPPrepareResult(
            inference_enabled=False,
            mode=packet.dcn_plan.frontier_plan.mode or "dcn_packet_prepare",
            prompt=prompt,
            frontier_prompt=frontier_prompt,
            frontier_max_tokens=req.frontier_max_tokens,
            token_budget=self._token_budget(req=req, input_tokens=frontier_tokens),
            dcn_packet_id=packet.packet_id,
            dcn_policy_version=packet.dcn_plan.policy_version,
            dml_context_used=bool(packet.dml_context),
            telemetry={
                "frontier_input_tokens": frontier_tokens,
                "packet_version": packet.packet_version,
                "inference_enabled": False,
            },
            context_observation=dict(packet.context_observation or {}),
            context_packet=dict(packet.context_packet or {}),
        )

    def _prepare_from_prompt(self, req: DIPPrepareRequest) -> DIPPrepareResult:
        if self.adapter is None:
            prompt = req.prompt
            frontier_prompt = (
                "Prepare this request for a frontier model. No DML adapter was configured, "
                "so no memory retrieval was performed.\n\n"
                f"User request:\n{prompt}\n\nFinal answer:"
            )
            tokens = utils.estimate_tokens(frontier_prompt)
            return DIPPrepareResult(
                inference_enabled=False,
                mode="frontier_full",
                prompt=prompt,
                frontier_prompt=frontier_prompt,
                frontier_max_tokens=req.frontier_max_tokens,
                token_budget=self._token_budget(req=req, input_tokens=tokens),
                dml_context_used=False,
                telemetry={"frontier_input_tokens": tokens, "inference_enabled": False},
                warnings=["no_dml_adapter_configured"],
            )

        pipeline = FrontierCompressionPipeline(
            self.adapter,
            config=FrontierPipelineConfig(
                top_k=req.top_k,
                local_max_tokens=req.local_max_tokens,
                frontier_max_tokens=req.frontier_max_tokens,
                include_local_draft=req.include_local_draft,
            ),
            draft_generator=self.draft_generator,
        )
        prepared = pipeline.prepare(
            req.prompt,
            tenant_id=req.scope.tenant_id,
            client_id=req.scope.client_id,
            session_id=req.scope.session_id,
            instance_id=req.scope.instance_id,
            top_k=req.top_k,
            local_max_tokens=req.local_max_tokens,
            frontier_max_tokens=req.frontier_max_tokens,
            include_local_draft=req.include_local_draft,
            direct_input_tokens_estimate=req.direct_input_tokens_estimate,
        )
        tokens = int((prepared.get("telemetry") or {}).get("frontier_input_tokens") or utils.estimate_tokens(prepared.get("frontier_prompt") or ""))
        return DIPPrepareResult(
            inference_enabled=False,
            mode=str(prepared.get("mode") or "prepare_only"),
            prompt=str(prepared.get("prompt") or req.prompt),
            frontier_prompt=str(prepared.get("frontier_prompt") or ""),
            frontier_max_tokens=int(prepared.get("frontier_max_tokens") or req.frontier_max_tokens),
            token_budget=self._token_budget(req=req, input_tokens=tokens),
            dml_context_used=bool(prepared.get("dml_context")),
            local_draft=str(prepared.get("local_draft") or ""),
            telemetry={**dict(prepared.get("telemetry") or {}), "inference_enabled": False},
        )

    def _attach_context_observation(self, req: DIPPrepareRequest, result: DIPPrepareResult) -> DIPPrepareResult:
        if self.context_controller is None:
            return result
        packet = req.cognitive_packet
        try:
            observation = self.context_controller.observe(
                scope=packet.scope if packet is not None else req.scope,
                model_limits={"context_window_tokens": result.token_budget.limit_tokens},
                current_prompt=result.frontier_prompt,
                current_messages=[{"role": "user", "content": result.frontier_prompt}],
                dcn_plan=packet.dcn_plan.to_dict() if packet is not None else None,
                dcn_packet=packet.to_dict() if packet is not None else None,
                dml_context=packet.dml_context if packet is not None else {},
                dpm_overlay=packet.dpm_overlay if packet is not None else {},
                output_reservation=result.frontier_max_tokens,
                runtime_reserved_tokens=req.runtime_reserved_tokens,
            )
        except Exception as exc:  # observe-only integration must not break DIP preparation
            return self._with_observation_warning(result, "context_observation_failed", exc)
        if not isinstance(observation, dict):
            return self._with_observation_warning(result, "context_observation_invalid")
        result.context_observation = dict(observation)
        if packet is not None:
            result.context_packet = dict(packet.context_packet or {})
        result.telemetry = {
            **dict(result.telemetry or {}),
            "context_pressure_state": (observation.get("pressure_state") or {}).get("state"),
            "context_capabilities": dict(observation.get("capabilities") or {}),
        }
        return result

    @staticmethod
    def _token_budget(*, req: DIPPrepareRequest, input_tokens: int) -> TokenBudget:
        reserved_tokens = req.frontier_max_tokens + req.runtime_reserved_tokens
        limit_tokens = req.model_context_tokens
        if limit_tokens is None:
            limit_tokens = input_tokens + reserved_tokens
        elif input_tokens + reserved_tokens > limit_tokens:
            raise ContractError("model_context_tokens cannot fit input tokens plus output/runtime reservations")
        return TokenBudget(limit_tokens=limit_tokens, used_tokens=input_tokens, reserved_tokens=reserved_tokens)

    @staticmethod
    def _with_observation_warning(
        result: DIPPrepareResult,
        reason: str,
        exc: Optional[Exception] = None,
    ) -> DIPPrepareResult:
        telemetry = {**dict(result.telemetry or {}), "context_observation_warning": reason}
        if exc is not None:
            telemetry["context_observation_error_type"] = type(exc).__name__
        result.telemetry = telemetry
        result.warnings = [*list(result.warnings or []), reason]
        return result

    @staticmethod
    def _prompt_from_packet(packet: CognitivePacket) -> str:
        return packet.assembled_context or str(packet.dcn_plan.to_dict())

    @staticmethod
    def _frontier_prompt_from_packet(*, packet: CognitivePacket, prompt: str) -> str:
        return (
            "You are the frontier reasoning layer for the Daystrom Platform. "
            "Use the DCN cognitive packet as structured control context. Do not infer beyond the packet's evidence.\n\n"
            f"DCN plan:\n{packet.dcn_plan.to_json()}\n\n"
            f"DPM overlay:\n{packet.dpm_overlay}\n\n"
            f"DML context:\n{packet.dml_context}\n\n"
            f"Prepared request:\n{prompt}\n\nFinal answer:"
        )


# Compatibility alias: the old frontier pipeline remains the implementation used
# for prompt+DML preparation, while this class gives the prototype DIP a named
# boundary.
DIPPreparationPipeline = InferencePreparationPipeline
