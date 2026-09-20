import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from token_meter.contracts import (
    DeletionDisposition,
    DetailLevel,
    DiscoveryContext,
    EvidenceBasis,
    SourceLocator,
    session_source_public_dict,
)
from token_meter.runtimes.omp import OMPRuntimeAdapter


class OMPRuntimeAdapterTests(unittest.TestCase):
    def _write_session(self, agent_dir, *, with_title=True, with_cost=True,
                       provider="anthropic", model="claude-test",
                       reasoning_tokens=None, model_usage=None,
                       nested_advisor=False):
        path = Path(agent_dir) / "sessions" / "--repo--" / "omp-session.jsonl"
        path.parent.mkdir(parents=True)
        usage = {
            "input": 100, "output": 20, "cacheRead": 10,
            "cacheWrite": 5, "totalTokens": 135,
        }
        if reasoning_tokens is not None:
            usage["reasoningTokens"] = reasoning_tokens
        if with_cost:
            usage["cost"] = {
                "input": 0.001, "output": 0.002,
                "cacheRead": 0.0001, "cacheWrite": 0.0002,
            }
        rows = []
        if with_title:
            rows.append({"type": "title", "v": 1, "title": "Fix the EUR header",
                         "source": "auto", "pad": " " * 32})
        rows.extend([
            {"type": "session", "version": 3, "id": "omp-session",
             "timestamp": "2026-09-19T10:00:00Z", "cwd": "/repo",
             "title": "Fix the EUR header"},
            {"type": "model_change", "id": "model-change", "parentId": None,
             "timestamp": "2026-09-19T10:00:01Z",
             "model": "{}/{}".format(provider, model)},
            {"type": "message", "id": "user", "parentId": "model-change",
             "timestamp": "2026-09-19T10:00:02Z",
             "message": {"role": "user", "content": []}},
            {"type": "message", "id": "assistant", "parentId": "user",
             "timestamp": "2026-09-19T10:00:05Z", "message": {
                "role": "assistant", "provider": provider, "model": model,
                "content": [{"type": "toolCall", "id": "call", "name": "read",
                             "arguments": {"path": "/repo/secret-plan"}}],
                "usage": usage,
             }},
            {"type": "message", "id": "result", "parentId": "assistant",
             "timestamp": "2026-09-19T10:00:06Z",
             "message": {"role": "toolResult", "toolCallId": "call", "toolName": "read"}},
        ])
        if model_usage is not None:
            rows.append(model_usage)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        if nested_advisor:
            advisor = path.parent / "omp-session" / "__advisor.jsonl"
            advisor.parent.mkdir(parents=True)
            advisor.write_text("\n".join(json.dumps(row) for row in [
                {"type": "title", "v": 1, "title": "", "pad": " " * 32},
                {"type": "session", "version": 3, "id": "advisor-session",
                 "timestamp": "2026-09-19T10:00:00Z", "cwd": "/repo"},
                {"type": "message", "id": "advisor-1",
                 "timestamp": "2026-09-19T10:00:09Z", "message": {
                    "role": "assistant", "provider": provider, "model": model,
                    "content": [],
                    "usage": {"input": 999, "output": 999, "cacheRead": 0,
                              "cacheWrite": 0, "totalTokens": 1998},
                 }},
            ]) + "\n")
        return path

    @staticmethod
    def _model_usage_row(purpose="auto-thinking", provider="openai", model="gpt-judge",
                         input_tokens=7, output_tokens=3):
        return {
            "type": "model_usage", "id": "aux-1", "parentId": "assistant",
            "timestamp": "2026-09-19T10:00:07Z",
            "purpose": purpose, "role": "tiny", "api": "openai-completions",
            "provider": provider, "model": model, "stopReason": "stop",
            "usage": {
                "input": input_tokens, "output": output_tokens,
                "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": input_tokens + output_tokens,
                "cost": {"input": 0.0007, "output": 0.0003,
                         "cacheRead": 0.0, "cacheWrite": 0.0},
            },
        }

    def test_skips_the_title_preamble_and_normalizes_measured_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp)
            adapter = OMPRuntimeAdapter(tmp)
            source = adapter.discover(DiscoveryContext(home="/home/test"))[0]
            loaded = adapter.load(source, DetailLevel.FULL)

        self.assertEqual(source.runtime_id, "omp")
        self.assertEqual(source.client_id, "omp")
        self.assertEqual(source.display_label, "Oh My Pi")
        self.assertEqual(source.session_id, "omp-session")
        self.assertEqual(source.model_ref.provider_id, "anthropic")
        self.assertEqual(source.model_ref.model_id, "claude-test")
        self.assertEqual(loaded.usage.input_tokens.value, 100)
        self.assertEqual(loaded.usage.output_tokens.value, 20)
        self.assertEqual(loaded.usage.cache_read_tokens.value, 10)
        self.assertEqual(loaded.usage.cache_write_tokens.value, 5)
        self.assertAlmostEqual(loaded.usage.cost_usd.value, 0.0033)
        self.assertEqual(loaded.usage.cost_usd.basis, EvidenceBasis.ESTIMATED)
        self.assertEqual([(tool.name, tool.status) for tool in loaded.tools], [("read", "success")])
        self.assertNotIn("locator", session_source_public_dict(source))
        self.assertNotIn("secret-plan", repr(loaded))

    def test_accepts_a_session_that_omits_the_title_preamble(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp, with_title=False)
            adapter = OMPRuntimeAdapter(tmp)
            sources = adapter.discover(DiscoveryContext(home="/home/test"))

        self.assertEqual([source.session_id for source in sources], ["omp-session"])

    def test_rejects_a_title_preamble_without_a_session_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions" / "--repo--" / "broken.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("\n".join(json.dumps(row) for row in [
                {"type": "title", "v": 1, "title": "orphan", "pad": " " * 32},
                {"type": "message", "timestamp": "2026-09-19T10:00:00Z",
                 "message": {"role": "assistant", "usage": {"input": 1, "output": 1}}},
            ]) + "\n")
            adapter = OMPRuntimeAdapter(tmp)

        self.assertEqual(adapter.discover(DiscoveryContext(home="/home/test")), ())

    def test_exposes_the_recorded_session_title_without_message_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp)
            adapter = OMPRuntimeAdapter(tmp)
            record = adapter.discover_legacy(DiscoveryContext(home="/home/test"))[0]

        self.assertEqual(record["provider"], "omp")
        self.assertEqual(record["label"], "Oh My Pi")
        self.assertEqual(record["runtime"], "Oh My Pi")
        self.assertEqual(record["title"], "Fix the EUR header")
        self.assertEqual(record["source_kind"], "omp_jsonl")

    def test_parses_the_combined_model_change_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_session(tmp, provider="alibaba-token-plan",
                                       model="qwen3.8-max")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            # Drop the assistant message so only model_change attribution remains.
            rows = [row for row in rows if row.get("id") != "assistant"]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            adapter = OMPRuntimeAdapter(tmp)
            record = adapter.discover_legacy(DiscoveryContext(home="/home/test"))[0]

        self.assertEqual(record["model"], "qwen3.8-max")
        self.assertEqual(record["model_provider"], "unknown-model-provider")

    def test_records_reasoning_tokens_as_an_output_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_session(tmp, reasoning_tokens=999)
            adapter = OMPRuntimeAdapter(tmp)
            turns = adapter._parsed(str(path))["turns"]

        self.assertEqual(turns[0]["reasoning_tokens"], 20)
        self.assertEqual(turns[0]["output_tokens"], 20)

    def test_counts_auxiliary_model_usage_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_session(tmp, model_usage=self._model_usage_row())
            adapter = OMPRuntimeAdapter(tmp)
            source = adapter.discover(DiscoveryContext(home="/home/test"))[0]
            loaded = adapter.load(source, DetailLevel.FULL)
            turns = adapter._parsed(str(path))["turns"]

        # The auxiliary call is an independent API call: assistant 100 + aux 7,
        # counted once each, never doubled and never dropped.
        self.assertEqual(loaded.usage.input_tokens.value, 107)
        self.assertEqual(loaded.usage.output_tokens.value, 23)
        self.assertAlmostEqual(loaded.usage.cost_usd.value, 0.0043)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[1]["purpose"], "auto-thinking")
        self.assertEqual(turns[1]["tools"], [])

    def test_does_not_discover_nested_advisor_or_subagent_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp, nested_advisor=True)
            adapter = OMPRuntimeAdapter(tmp)
            sources = adapter.discover(DiscoveryContext(home="/home/test"))
            loaded = adapter.load(sources[0], DetailLevel.FULL)

        self.assertEqual([source.session_id for source in sources], ["omp-session"])
        # The advisor transcript's 999-token call must not leak into the main
        # session totals through any path.
        self.assertEqual(loaded.usage.input_tokens.value, 100)

    def test_redacts_account_bearing_bedrock_profile_from_native_model_identity(self):
        raw_profile = (
            "arn:aws:bedrock:us-west-2:123456789012:"
            "application-inference-profile/private-profile"
        )
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp, provider="amazon-bedrock", model=raw_profile)
            adapter = OMPRuntimeAdapter(tmp)
            source = adapter.discover(DiscoveryContext(home="/home/test"))[0]
            loaded = adapter.load(source, DetailLevel.SUMMARY)

        self.assertEqual(source.model_ref.provider_id, "amazon")
        self.assertEqual(source.model_ref.model_id, "aws-bedrock-profile")
        self.assertNotIn(raw_profile, repr(source))
        self.assertNotIn("123456789012", repr(loaded))

    def test_missing_omp_cost_is_unavailable_without_changing_measured_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp, with_cost=False)
            adapter = OMPRuntimeAdapter(tmp)
            source = adapter.discover(DiscoveryContext(home="/home/test"))[0]
            loaded = adapter.load(source, DetailLevel.SUMMARY)

        self.assertEqual(loaded.usage.input_tokens.value, 100)
        self.assertEqual(loaded.usage.cost_usd.basis, EvidenceBasis.UNAVAILABLE)
        self.assertIsNone(loaded.usage.cost_usd.value)
        self.assertEqual(loaded.turns, ())

    def test_deletion_plan_trashes_owned_traces_and_denies_foreign_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_session(tmp)
            adapter = OMPRuntimeAdapter(tmp)
            source = adapter.discover(DiscoveryContext(home="/home/test"))[0]
            plan = adapter.deletion_plan(source)
            foreign = replace(source, locator=SourceLocator("jsonl", "/etc/omp-session.jsonl"))
            denied_foreign = adapter.deletion_plan(foreign)
            denied_shape = adapter.deletion_plan({"path": str(source.locator.value)})

        self.assertEqual(plan.disposition, DeletionDisposition.TRASH)
        self.assertEqual(plan.targets, (source.locator,))
        self.assertEqual(denied_foreign.disposition, DeletionDisposition.DENY)
        self.assertEqual(denied_shape.disposition, DeletionDisposition.DENY)


if __name__ == "__main__":
    unittest.main()
