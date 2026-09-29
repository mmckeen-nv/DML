# M7 closed — Llama 3 SFT v3

The separately registered V3 candidate passed the original frozen campaign **9/9**, including `supersede_then_answer` and `read_both_commits`. Primary and independent data replays passed every original M7 gate and agreed on the complete parsed evidence. Independent review approved **M7 closure only**; `production_ready=false`, and nothing is merged.

Exact source `29c7de2195c266adaca8139375df742cc867694b` passed all **22 CI jobs**, including 112 SFT and 1,589 model-input controls with zero skips in those lanes. The registered consumer matched all 774 retained prompt/token inputs. GPU admission passed one fixed nonsampled forward, and fresh qualification passed **2/2 planning and 4/4 readiness**, followed by both evidence replays.

The campaign used the fixed final SFT v3 adapter on the original pinned Meta Llama 3 8B Instruct base, running directly on the station's GB300. It ran once: **21 model calls, 27,773 input tokens, 782 output tokens**, with zero execution errors, rejected actions, timeouts, or unknown usage/effects. All original corpus, retrieval, authorization, claims/citations, sampling, and budget rules were preserved. No retries, output repair, or forced completion occurred. The original gates do not require all nine successes; this run nevertheless achieved nine.

Both replays checked all 2,170 frozen files and preserved 87 retained result files. Different JSON serialization produces different evidence hashes; complete parsed evidence is equal. Replay validates retained token, grammar, authority, and outcome evidence without regenerating logits. TTFT is unmeasured, and repeated-error rate is null because there were zero error opportunities. This single campaign does not establish broad model reliability or production readiness.

Raw evidence remains at `/home/nvidia/dml-debug/llama3-sft-v3-integrated-v1-20260929`. The [compact outcome and hashes](artifacts/llama3-sft-v3-m7-outcome-2026-09-29.json) records the model revision, adapter, tokenizer/template, runtime configuration, sampling, limits, CI, qualification, campaign, and review bindings. Rejected preparation drafts and all earlier failures remain retained. The historical Nemotron 8/9 campaign is unchanged.

- Exact-source CI: [36627875864](https://github.com/mmckeen-nv/DML/actions/runs/36627875864).
- Campaign SHA256: `9576b0438baef80a92f8dc28bca61a35a3f21c455d8e384b4477a1474fe0d480`.
- Independent closure review SHA256: `bc0852bb2702df462c1646fac4730e9d2702e35e9e9275185aefb93a343c9063`.
- Compact outcome SHA256: `8681fee8365800e6782934c3f8d58b5a37c2ea2023e712e4c43870263e117359`.

Milestones **M1–7 are closed**, **M8–11 remain open**, and **M12–13 remain deferred**. The next release milestone is M8's fair baseline comparison; this closure does not authorize its execution.
