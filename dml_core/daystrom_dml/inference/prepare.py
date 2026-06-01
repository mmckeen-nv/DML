"""DIP preparation boundary around the legacy frontier pipeline."""
from __future__ import annotations

from typing import Any, Optional

from daystrom_dml import utils
from daystrom_dml.api_contracts import DaystromScope, TokenBudget
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
    ) -> None:
        self.adapter = adapter
        self.config = config or FrontierPipelineConfig()
        self.draft_generator = draft_generator

    def prepare(self, request: DIPPrepareRequest | dict[str, Any]) -> DIPPrepareResult:
        req = request if isinstance(request, DIPPrepareRequest) else DIPPrepareRequest.from_dict(request)
        if req.cognitive_packet is not None:
            return self._prepare_from_packet(req, req.cognitive_packet)
        return self._prepare_from_prompt(req)

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
            token_budget=TokenBudget(limit_tokens=max(req.frontier_max_tokens, frontier_tokens), used_tokens=frontier_tokens),
            dcn_packet_id=packet.packet_id,
            dcn_policy_version=packet.dcn_plan.policy_version,
            dml_context_used=bool(packet.dml_context),
            telemetry={
                "frontier_input_tokens": frontier_tokens,
                "packet_version": packet.packet_version,
                "inference_enabled": False,
            },
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
                token_budget=TokenBudget(limit_tokens=max(req.frontier_max_tokens, tokens), used_tokens=tokens),
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
            token_budget=TokenBudget(limit_tokens=max(req.frontier_max_tokens, tokens), used_tokens=tokens),
            dml_context_used=bool(prepared.get("dml_context")),
            local_draft=str(prepared.get("local_draft") or ""),
            telemetry={**dict(prepared.get("telemetry") or {}), "inference_enabled": False},
        )

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
