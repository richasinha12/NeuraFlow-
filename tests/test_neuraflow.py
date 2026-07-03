"""
Tests for NeuraFlow. Covers shared services + all five features.
Run:  pytest tests/test_neuraflow.py -v --asyncio-mode=auto
"""
import sys, os, pytest, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neuraflow import (
    cfg, _pii, _chunker, _vstore, _cache, _obs, _llm, _log,
    _faithfulness, _relevance, _context_precision, _context_recall,
    _format_sample, _qlora_config,
    rag_ingest, rag_query, run_agents, finetune_start,
    run_evaluation, get_dashboard, get_logs,
    RequestLog, AgentTools,
)


# shared services ────────────────────────────────────────────────────────────

class TestPII:
    def test_email(self):
        out, found = _pii.redact("Email: user@example.com")
        assert "EMAIL" in found and "user@example.com" not in out
    def test_phone(self):
        _, f = _pii.redact("Call 555-123-4567")
        assert "PHONE" in f
    def test_ssn(self):
        _, f = _pii.redact("SSN 123-45-6789")
        assert "SSN" in f
    def test_no_pii(self):
        out, f = _pii.redact("What is the refund policy?")
        assert out == "What is the refund policy?" and f == []
    def test_multiple(self):
        _, f = _pii.redact("test@x.com and 555-000-1111")
        assert len(f) >= 2


class TestChunker:
    def test_fixed_creates_chunks(self):
        chunks = _chunker.fixed_size(" ".join(["word"]*600))
        assert len(chunks) >= 2
    def test_semantic_preserves_content(self):
        text  = "Para A.\n\nPara B.\n\nPara C."
        combo = " ".join(c["text"] for c in _chunker.semantic(text))
        assert "Para A" in combo and "Para C" in combo
    def test_hierarchical_has_summary(self):
        chunks = _chunker.hierarchical(" ".join(["w"]*3500), title="Doc")
        assert any(c.get("chunk_type") == "summary" for c in chunks)
    def test_auto_short(self):
        _, s = _chunker.auto(" ".join(["w"]*100))
        assert s == "fixed_size"
    def test_auto_long(self):
        _, s = _chunker.auto(" ".join(["w"]*4000))
        assert s == "hierarchical"


class TestVectorStore:
    def test_upsert_increases_count(self):
        before = _vstore.count("vs_a")
        _vstore.upsert("vs_a", [{"text":"hello","idx":0,"strategy":"fixed_size"}], "f.txt")
        assert _vstore.count("vs_a") > before
    def test_search_demo_fallback(self):
        res = _vstore.search("empty_xyz", "anything", top_k=3)
        assert len(res) >= 1
    def test_rerank_top_k(self):
        cands = [{"text":f"doc {i}","score":0.5,"idx":i} for i in range(10)]
        assert len(_vstore.rerank("q", cands, top_k=3)) == 3
    def test_delete(self):
        _vstore.upsert("del", [{"text":"tmp","idx":0,"strategy":"fixed_size"}], "x.txt")
        d = _vstore.delete("del")
        assert d > 0 and _vstore.count("del") == 0


class TestCache:
    def test_miss(self):
        assert _cache.get("definitely-not-cached-xyz") is None
    def test_set_get(self):
        _cache.set("my query", {"answer": "yes"})
        assert _cache.get("my query")["answer"] == "yes"
    def test_case_insensitive(self):
        _cache.set("Hello", {"answer": "hi"})
        assert _cache.get("HELLO") is not None


class TestObservability:
    def test_record_and_query(self):
        _log("obs_t","rag","gpt-4o-mini",500,200,840.0,0.91,False)
        assert len(_obs.query("obs_t",1)) >= 1
    def test_cost(self):
        log = RequestLog("r1","t","rag","gpt-4o-mini",1000,500,500)
        expected = (1000/1000)*0.00015 + (500/1000)*0.0006
        assert abs(log.cost_usd - expected) < 0.0001
    def test_unknown_model_zero_cost(self):
        assert RequestLog("r2","t","rag","unknown-xyz",100,100,100).cost_usd == 0.0


# feature 1 — rag ─────────────────────────────────────────────────────────────

class TestRAGMetrics:
    def test_faithfulness_range(self):
        s = _faithfulness("answer text", ["answer context text"])
        assert 0.0 <= s <= 1.0
    def test_faithfulness_empty(self):
        assert _faithfulness("answer", []) == 0.0
    def test_relevance_range(self):
        s = _relevance("What is RAG?", "RAG is retrieval augmented generation")
        assert 0.0 <= s <= 1.0


class TestRAGIngest:
    @pytest.mark.asyncio
    async def test_ingest_returns_chunks(self):
        r = await rag_ingest("ri_t","doc.txt",b"Hello world AI.",512)
        assert r["chunks"] >= 1 and r["file"] == "doc.txt"
    @pytest.mark.asyncio
    async def test_ingest_updates_store(self):
        before = _vstore.count("ri_c")
        await rag_ingest("ri_c","f.txt",b"New AI content here for testing.",512)
        assert _vstore.count("ri_c") > before


class TestRAGQuery:
    @pytest.mark.asyncio
    async def test_required_fields(self):
        r = await rag_query("rq_t","What is ML?",3,False)
        for f in ["answer","sources","faithfulness","latency_ms","cached"]:
            assert f in r
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        q = f"cache test {uuid.uuid4()}"
        r1 = await rag_query("ct",""+q,3,True)
        r2 = await rag_query("ct",""+q,3,True)
        assert not r1["cached"] and r2["cached"]
    @pytest.mark.asyncio
    async def test_pii_detected(self):
        r = await rag_query("rq_p","My email is x@y.com, what is the policy?",3,False)
        assert r["pii_redacted"] is True


# feature 2 — agents ──────────────────────────────────────────────────────────

class TestAgentTools:
    @pytest.mark.asyncio
    async def test_web_search(self):
        r = await AgentTools.web_search("AI 2026")
        assert r["tool"] == "web_search"
    def test_calculator(self):
        assert AgentTools.calculator("3 * 7")["result"] == 21
    def test_calculator_error(self):
        assert "error" in AgentTools.calculator("import os")


class TestAgents:
    @pytest.mark.asyncio
    async def test_required_fields(self):
        r = await run_agents("Research AI","ag_t",["web_search"],False)
        for f in ["run_id","status","final_output","trace","latency_ms"]:
            assert f in r
    @pytest.mark.asyncio
    async def test_all_nodes_run(self):
        r = await run_agents("Write about ML","ag_t2",[],False)
        trace = " ".join(r["trace"])
        for n in ["planner","researcher","writer","reviewer"]:
            assert n in trace
    @pytest.mark.asyncio
    async def test_hitl_recorded(self):
        r = await run_agents("task","ag_h",[],True)
        assert r["hitl_used"] and any("hitl" in s for s in r["trace"])
    @pytest.mark.asyncio
    async def test_unique_run_ids(self):
        r1 = await run_agents("t1","ag_u",[],False)
        r2 = await run_agents("t2","ag_u",[],False)
        assert r1["run_id"] != r2["run_id"]


# feature 3 — fine-tuning ─────────────────────────────────────────────────────

class TestFineTuning:
    def test_format_alpaca(self):
        s = {"instruction":"Classify:","input":"Good!","output":"Positive"}
        out = _format_sample(s,"alpaca")
        assert "Classify:" in out and "Positive" in out
    def test_qlora_has_lora(self):
        c = _qlora_config("meta-llama/Llama-3-8b-hf",16,3)
        assert c["lora"]["r"] == 16
    def test_qlora_4bit(self):
        c = _qlora_config("meta-llama/Llama-3-8b-hf",16,3)
        assert c["quantization"]["load_in_4bit"] is True
    @pytest.mark.asyncio
    async def test_start_returns_job(self):
        r = await finetune_start("ft_t",[],cfg.BASE_MODEL,16,2)
        assert r["status"] == "queued" and r["job_id"].startswith("ft_")
    @pytest.mark.asyncio
    async def test_custom_dataset(self):
        data = [{"instruction":"Q:","input":"t","output":"a"},
                {"instruction":"Q:","input":"t2","output":"b"}]
        r = await finetune_start("ft_t2",data,cfg.BASE_MODEL,8,1)
        assert r["total_samples"] == 2


# feature 4 — evaluation ──────────────────────────────────────────────────────

class TestEvaluation:
    @pytest.mark.asyncio
    async def test_all_metrics(self):
        r = await run_evaluation("ev_t",[])
        for m in ["faithfulness","answer_relevance","context_precision",
                  "context_recall","overall"]:
            assert m in r["metrics"]
    @pytest.mark.asyncio
    async def test_in_range(self):
        r = await run_evaluation("ev_t2",[])
        assert all(0.0 <= v <= 1.0 for v in r["metrics"].values())
    @pytest.mark.asyncio
    async def test_ci_gate_string(self):
        r = await run_evaluation("ev_t3",[])
        assert "passed" in r["ci_gate"] or "failed" in r["ci_gate"]
    @pytest.mark.asyncio
    async def test_custom_dataset(self):
        data = [{"question":"Q?","answer":"A","contexts":["C"],"ground_truth":"G"}]
        r = await run_evaluation("ev_c",data)
        assert r["total_samples"] == 1


# feature 5 — monitoring ──────────────────────────────────────────────────────

class TestMonitoring:
    @pytest.mark.asyncio
    async def test_dashboard_sections(self):
        r = await get_dashboard("dash_t",24)
        for k in ["latency","quality","cost","drift"]:
            assert k in r
    @pytest.mark.asyncio
    async def test_logs_endpoint(self):
        _log("log_t","rag","gpt-4o-mini",100,50,500,0.9)
        r = await get_logs("log_t",10)
        assert "logs" in r and r["count"] >= 1


# integration ─────────────────────────────────────────────────────────────────

class TestIntegration:
    @pytest.mark.asyncio
    async def test_rag_then_agent_shares_store(self):
        await rag_ingest("int_t","p.txt",b"Return policy: 30-day refund.",512)
        r = await run_agents("Find return policy","int_t",["rag_retriever"],False)
        assert r["status"] in ["completed","needs_revision"]

    @pytest.mark.asyncio
    async def test_all_features_log_to_same_store(self):
        tid = f"int_{uuid.uuid4().hex[:6]}"
        before = len(_obs.query(tid,24))
        await rag_query(tid,"test",3,False)
        await run_agents("task",tid,[],False)
        await run_evaluation(tid,[])
        await finetune_start(tid,[],cfg.BASE_MODEL,8,1)
        assert len(_obs.query(tid,24)) >= before + 4

    @pytest.mark.asyncio
    async def test_dashboard_after_requests(self):
        tid = f"dash_{uuid.uuid4().hex[:6]}"
        await rag_query(tid,"question",3,False)
        await run_agents("task",tid,[],False)
        r = await get_dashboard(tid,24)
        assert r["total_requests"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "--tb=short"])
