import io
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from urllib.parse import quote

from web import server


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp.name) / "runs.db"
        cls.db_patch = patch.object(server, "DB_PATH", cls.db_path)
        cls.db_patch.start()
        server.initialize()
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close(); cls.db_patch.stop(); cls.temp.cleanup()

    def get_json(self, path):
        with urlopen(self.base + path) as response:
            return json.load(response)

    def test_economy_cards_discovery_queue_and_dispatch(self):
        cases = {item["id"]: item for item in self.get_json("/api/cases")}
        models = {"deepseek": "deepseek-v4-pro", "glm": "glm-5.3", "qwen": "qwen3.7-max",
                  "kvv": "kimi-k3", "fable": "claude-sonnet-4-6", "gpt": "gpt-6.1-sol"}
        for kind, model in models.items():
            card = f"{kind.upper()}_ECONOMY"
            protocol = "anthropic" if kind == "fable" else "openai"
            self.assertTrue(cases[card]["economy_validation"])
            self.assertEqual(cases[card]["requests"], 11)
            payload = {"test_kind": kind, "case": card, "protocol": protocol,
                       "base_url": "https://example.test", "api_key": "secret", "model": model}
            with patch.object(server.JOB_QUEUE, "put") as put:
                with urlopen(Request(self.base + "/api/runs", data=json.dumps(payload).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})) as response:
                    result = json.load(response)
            self.assertEqual(result["case_title"], cases[card]["title"])
            self.assertEqual(put.call_args.args[0]["case"], card)
            captured = self._execute_child_capture({"test_kind": kind, "case": card,
                                                   "protocol": protocol, "model": model})
            self.assertEqual(Path(captured["command"][1]).name, "run_economy_validation.py")
            self.assertEqual(captured["command"][2], card)
            self.assertEqual(captured["env"]["VOLC_MODEL"], model)
            self.assertNotIn("KVV_TOOL_SCHEMA", captured["command"])

    def test_economy_scope_and_invalid_configuration(self):
        payload = {"test_kind": "gpt", "case": "GPT_ECONOMY", "protocol": "openai",
                   "base_url": "https://example.test", "api_key": "secret", "model": "gpt-6.1-sol",
                   "case_models": ["gpt-6.1-sol", "gpt-6-sol"]}
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=json.dumps(payload).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        self.assertEqual([child["model"] for child in put.call_args.args[0]["children"]], payload["case_models"])
        for override in ({"case_models": ["gpt-6.1-sol", "kimi-k3"]}, {"faces": ["model"]},
                         {"model": "kimi-k3"}, {"protocol": "anthropic"}):
            with patch.object(server.JOB_QUEUE, "put") as put:
                with self.assertRaises(HTTPError) as caught:
                    urlopen(Request(self.base + "/api/runs", data=json.dumps({**payload, **override}).encode(),
                                    method="POST", headers={"Content-Type": "application/json"}))
                self.assertEqual(caught.exception.code, 400)
                put.assert_not_called()

    def test_economy_and_scoped_specialty_keep_all_models(self):
        for kind, models, cards, protocol in (
            ("qwen", ["qwen3.7-max", "qwen3.8-max"], ["QWEN_ECONOMY", "QWEN_TIER"], "openai"),
            ("fable", ["claude-opus-5", "claude-sonnet-5"], ["FABLE_ECONOMY", server.PROTO_SIGNATURE_CASE_ID], "anthropic"),
        ):
            payload = {"test_kind": kind, "cases": cards, "protocol": protocol,
                       "base_url": "https://example.test", "api_key": "secret", "model": models[0],
                       "case_models": models, "signature_max_tokens": 4096}
            with patch.object(server.JOB_QUEUE, "put") as put:
                with urlopen(Request(self.base + "/api/runs", data=json.dumps(payload).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})) as response:
                    json.load(response)
            children = put.call_args.args[0]["children"]
            self.assertEqual({(c["case"], c["model"]) for c in children}, {(c, m) for c in cards for m in models})
            if kind == "fable":
                self.assertTrue(all(c["signature_max_tokens"] == 4096 for c in children if c["case"] == server.PROTO_SIGNATURE_CASE_ID))

    def test_kimi_schema_can_be_selected_without_channel_toggle(self):
        card = next(c for c in self.get_json("/api/cases") if c["id"] == "KVV_TOOL_SCHEMA")
        self.assertTrue(card["manual_execution"])
        self.assertTrue(card["scoring_excluded"])
        self.assertEqual(card["requests"], 408)
        payload = {"test_kind": "kvv", "case": "KVV_TOOL_SCHEMA", "protocol": "openai",
                   "base_url": "https://example.test", "api_key": "secret", "model": "kimi-k3"}
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=json.dumps(payload).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "KVV_TOOL_SCHEMA")
        job = put.call_args.args[0]
        self.assertEqual(job["faces"], "model,verify")
        captured = self._execute_child_capture(job)
        self.assertEqual(Path(captured["command"][1]).name, "run_kvv.py")
        self.assertEqual(captured["command"][2], "KVV_TOOL_SCHEMA")

    def test_run_table_carries_scoring_columns(self):
        """评分三列靠迁移建出来：列表页要读 score/score_grade/score_coverage，缺列直接 500。"""
        with sqlite3.connect(self.db_path) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        self.assertTrue({"score", "score_grade", "score_coverage"} <= columns)

    def test_latest_result_finds_temporary_suite_for_single_case(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            suite_result = results / "K3_TEMP_SUITE-20260922-172637"
            suite_result.mkdir()
            with patch.object(server, "RESULTS", results):
                # 与 web/server.py 收尾阶段的真实调用一致：单案例执行时
                # 产物目录是 K3_TEMP_SUITE-*，通过 aliases 命中。
                found = server.latest_result(
                    "CACHE-HIT-7", time.time() - 2, aliases=("K3_TEMP_SUITE",)
                )
            self.assertEqual(found, suite_result)

    def test_worker_failure_marks_job_failed_instead_of_dying_silently(self):
        run_id = f"worker-failure-{time.time_ns()}"
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, "load", "T0", "openai", "https://example.com", "m", "queued", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
            )
        try:
            server._mark_worker_failure({"id": run_id}, RuntimeError("boom"))
            with server.db() as connection:
                row = connection.execute("SELECT status,error,exit_code FROM runs WHERE id=?", (run_id,)).fetchone()
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["exit_code"], -1)
            self.assertIn("RuntimeError: boom", row["error"])
        finally:
            with server.db() as connection:
                connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_single_kvv_temp_run_persists_selected_case_name(self):
        payload = json.dumps({
            "protocol": "volcengine", "test_kind": "kvv",
            "base_url": "https://example.com", "api_key": "secret", "model": "kimi-k3",
            "case": "K3_TEMP_SUITE", "temporary_case": "CACHE-HIT-7",
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "CACHE-HIT-7")
        self.assertEqual(result["case_title"], "90% 稳定前缀命中探测")
        self.assertEqual(put.call_args.args[0]["temporary_case"], "CACHE-HIT-7")

    def test_interrupted_image_summary_keeps_partial_request_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            rows = [
                {"success": True, "error_type": None},
                {"success": False, "error_type": "http_503"},
                {"success": False, "error_type": "request_timeout"},
            ]
            (result / "requests.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = server.interrupted_request_summary(
                {
                    "test_kind": "gpt-image",
                    "case": "GPT_IMAGE_2",
                    "protocol": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "gpt-image-2",
                    "requests": 5,
                },
                "run-id",
                result,
            )
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["unfinished"], 2)
        self.assertEqual(
            summary["error_distribution"],
            {"http_503": 1, "request_timeout": 1},
        )

    def test_cases_and_index(self):
        cases=self.get_json("/api/cases")
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual({x["id"] for x in cases if x["id"].endswith("CONTRACT")}, {"KVV_CONTRACT", "DS_CONTRACT", "GLM_CONTRACT"})
        self.assertEqual(next(x for x in cases if x["id"]=="KVV_CONTRACT")["test_kind"], "kvv")
        self.assertEqual(next(x for x in cases if x["id"]=="DS_CONTRACT")["test_kind"], "deepseek")
        self.assertFalse(any(x["test_kind"] == "strict" for x in cases))
        # 「千问测试」：单一档位 + 六款型号共用同一批用例；档位卡片开放多模型执行范围。
        qwen_cards = [x for x in cases if x["test_kind"] == "qwen"]
        self.assertEqual({x["id"] for x in qwen_cards},
                         {"QWEN_CORE", "QWEN_THINKING", "QWEN_AGENT", "QWEN_TIER", "QWEN_VISION", "QWEN_CAPABILITY", "QWEN_VERIFY", "QWEN_FULL", "QWEN_ECONOMY"})
        qwen_core_card = next(x for x in qwen_cards if x["id"] == "QWEN_CORE")
        self.assertEqual(qwen_core_card["default_model"], "qwen3.8-max")
        self.assertEqual([m["id"] for m in qwen_core_card["qwen_models"]],
                         ["qwen3.8-max", "qwen3.8-flash", "qwen3.7-max", "qwen3.7-plus",
                          "qwen3.6-plus", "qwen3.6-flash"])
        self.assertEqual(qwen_core_card["protocols"], ["openai"])
        self.assertFalse(qwen_core_card["model_scope"])
        self.assertTrue(next(x for x in qwen_cards if x["id"] == "QWEN_TIER")["model_scope"])
        load_ids={x["id"] for x in cases if x["test_kind"] == "load"}
        self.assertEqual(load_ids,{"T0","RANDOM_LOAD","B1","SIMPLE_LOAD","FULL_LOAD","C0","C1"})
        random_load=next(x for x in cases if x["id"]=="RANDOM_LOAD")
        self.assertEqual((random_load["concurrency"],random_load["requests"]),(10,100))
        self.assertEqual(random_load["output_tokens"],"4096")
        self.assertTrue(random_load["manual_parameters"])
        simple=next(x for x in cases if x["id"]=="SIMPLE_LOAD")
        full=next(x for x in cases if x["id"]=="FULL_LOAD")
        # L1–L5 用预设样本数、O1 用自己的档位样本数（2026-10-14：O1 50→100，故总数 +50）。
        self.assertEqual((simple["concurrency"],simple["requests"]),(30,350))
        self.assertEqual((full["concurrency"],full["requests"]),(50,1100))
        self.assertIn("L1–L5 各 50",simple["request_text"])
        self.assertIn("L1–L5 各 200",full["request_text"])
        self.assertIn("O1 100",full["request_text"])
        required={"concurrency_text","input_tokens","request_text","warmup","steady","admission"}
        self.assertTrue(all(required.issubset(item) for item in cases))
        b1=next(x for x in cases if x["id"]=="B1")
        self.assertEqual((b1["concurrency_text"],b1["request_text"]),("2","成功目标 16；最多尝试 16"))
        self.assertIn("2K / 8K / 16K / 32K",b1["input_tokens"])
        self.assertIn("L1–L5",simple["workload_profile"])
        self.assertTrue(simple["fixed_parameters"])
        # Seedance 卡片：显式参数面板所需的取值域必须由后端下发（前端不写死），
        # 且执行规模上限与 cases.py 的常量一致，否则前端预校验会放行服务端要拒的输入。
        seedance=next(x for x in cases if x["test_kind"]=="seedance")
        self.assertEqual(seedance["id"], "SEEDANCE_VIDEO_GEN")
        self.assertEqual(seedance["protocols"], ["seedance"])
        # 分辨率 / 图片位置 / 时长区间都必须跟该面官方文档对齐：480p 与「单图 last_frame」
        # 都不在 2.0/2.5 的可选集里——首跑（2026-10-13）就是拿 480p 打 2.0，提交 200 拿
        # task id、轮询 4.6 分钟后任务转 failed，文案「请求参数校验失败」。
        self.assertEqual(seedance["seedance_resolutions"], ["720p","1080p","4k"])
        self.assertEqual(seedance["seedance_ratios"], ["16:9","4:3","1:1","3:4","9:16","21:9","adaptive"])
        self.assertEqual([r["id"] for r in seedance["seedance_image_roles"]],
                         ["first_frame","reference_image"])
        # 时长上限给的是**并集**（2.5 的 30）；按档位收紧留给服务端入队校验。
        self.assertEqual((seedance["seedance_duration_min"],seedance["seedance_duration_max"]),
                         (4,30))
        self.assertEqual(seedance["seedance_default_resolution"], "720p")
        self.assertEqual(seedance["seedance_default_ratio"], "adaptive")
        self.assertEqual(seedance["seedance_model_ids"], ["Seedance-2.0","Seedance-2.5"])
        self.assertIn("Seedance-2.5", seedance["seedance_duration_note"])
        self.assertEqual(seedance["seedance_max_requests"], 100)
        self.assertEqual(seedance["seedance_max_concurrency"], 16)
        self.assertEqual((seedance["requests"],seedance["concurrency"]),(4,2))
        self.assertIn("blocked", seedance["admission"])
        protocols=self.get_json("/api/protocols")
        self.assertEqual(len(protocols), 10)
        self.assertEqual(next(x for x in protocols if x["id"]=="openai_responses")["path_template"], "/v1/responses")
        qianfan=next(x for x in protocols if x["id"]=="qianfan")
        self.assertEqual(qianfan["default_base_url"], "https://qianfan.baidubce.com")
        self.assertEqual(qianfan["path_template"], "/v2/chat/completions")
        # 千帆走 OpenAI 线格式，因此 family 必须是 "openai"：所有 openai 家族的套件靠它放行。
        self.assertEqual(qianfan["family"], "openai")
        gemini_images=next(x for x in protocols if x["id"]=="gemini_images")
        self.assertEqual(gemini_images["path_template"], "/v1beta/models/{model}:generateContent")
        self.assertEqual(gemini_images["test_kinds"], ["gemini-image"])
        qwen=next(x for x in protocols if x["id"]=="qwen")
        # 千问走 OpenAI 兼容模式：family 必须是 "openai"（共用行情/契约/对话套件），
        # api root 是 /compatible-mode/v1，且 test_kinds 必须放行 qwen 套件。
        self.assertEqual(qwen["family"], "openai")
        self.assertEqual(qwen["default_base_url"], "https://dashscope.aliyuncs.com")
        self.assertEqual(qwen["path_template"], "/compatible-mode/v1/chat/completions")
        self.assertIn("qwen", qwen["test_kinds"])
        seedance=next(x for x in protocols if x["id"]=="seedance")
        # 视频生成是异步任务面：根路径 /api/v3（不是 /v1），family 必须独立于 volcengine，
        # 否则 supports_test_kind 会把 chat 面也放行给 seedance。
        self.assertEqual(seedance["family"], "seedance")
        self.assertEqual(seedance["default_base_url"], "https://token-hub.tangrankeji.com")
        self.assertEqual(seedance["path_template"], "/api/v3/contents/generations/tasks")
        self.assertEqual(seedance["test_kinds"], ["seedance"])
        with urlopen(self.base + "/") as response:
            self.assertIn("性能测试台", response.read().decode())

    # 「无独立 tab」的后端 test_kind 白名单：被 open-source tab 分组收纳的，以及
    # 只保留标签、已无独立入口的历史档位。除此之外必须每个 test_kind 都有一个 tab。
    _GROUPED_TEST_KINDS = {"arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark"}
    _TABLESS_TEST_KINDS = {"authenticity"}
    # 前端合成的分组 tab，本身不是后端 test_kind。
    _SYNTHETIC_TABS = {"open-source"}
    # kindLabel 的默认分支就是「多轮压测」，因此 load 没有显式分支。
    _DEFAULT_TAB = "load"

    def test_every_test_kind_has_a_tab_and_every_tab_has_a_label(self):
        """前端 tab 与后端 test_kind 双向对齐 —— 新增套件时最容易漏的登记点。

        真实事故：插入「Gemini 测试」tab 时把相邻的 `data-kind="gemini-image"` 按钮一起替换掉，
        后端 `gemini_images` 协议仍放行 `gemini-image`，前端却再无入口（tab 静默消失）。
        没有断言守卫时这种替换不会报错，只能靠肉眼发现。
        """
        static = Path(server.ROOT) / "web" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "app.js").read_text(encoding="utf-8")
        tabs = set(re.findall(r'data-kind="([a-z-]+)"', html))
        self.assertTrue(tabs, "index.html 未解析到任何 data-kind tab（选择器可能已改动）")
        kinds = {kind for proto in self.get_json("/api/protocols") for kind in proto["test_kinds"]}
        expected = kinds - self._GROUPED_TEST_KINDS - self._TABLESS_TEST_KINDS
        self.assertEqual(tabs - self._SYNTHETIC_TABS, expected,
                         "前端 tab 与后端 test_kind 不再一一对应（多出或缺失）")
        # 文本面与图像面必须各有自己的 tab，且不可互相顶掉。
        self.assertIn("gemini", tabs)
        self.assertIn("gemini-image", tabs)
        # 视频面同理：Seedance 的 tab 不能被相邻的 gemini-image 按钮替换掉。
        self.assertIn("seedance", tabs)
        # 每个 tab 都要能在 kindLabel 里取到名字，否则界面会显示成默认文案。
        for kind in sorted(tabs - {self._DEFAULT_TAB}):
            self.assertIn(f"kind==='{kind}'?", js, f"kindLabel 缺少 {kind} 分支")
        # 刷新恢复的白名单必须包含所有 tab，否则切换后再刷新会静默跳回「多轮压测」。
        match = re.search(r"state\.testKind=\[(.*?)\]\.includes\(saved\.testKind\)", js)
        self.assertIsNotNone(match, "未找到 state.testKind 的恢复白名单")
        allowed = set(re.findall(r"'([a-z-]+)'", match.group(1)))
        self.assertTrue(tabs <= allowed, f"恢复白名单缺少：{sorted(tabs - allowed)}")

    def test_every_element_referenced_by_app_js_exists_in_index_html(self):
        """面板 id 写错不会报错，只会让事件绑不上去（静默失效）。

        新增套件最容易踩的两个坑：① index.html 里漏写控件；② app.js 里的 id 拼错。
        两者在浏览器里都表现为「点了没反应」，靠肉眼很难发现——静态对一遍 id 就够了。
        """
        static = Path(server.ROOT) / "web" / "static"
        js = (static / "app.js").read_text(encoding="utf-8")
        html = (static / "index.html").read_text(encoding="utf-8")
        declared = set(re.findall(r'id="([^"]+)"', html))
        referenced = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", js))
        self.assertTrue(referenced, "app.js 未解析到任何 $('#id') 引用（选择器写法可能已改动）")
        missing = sorted(referenced - declared)
        self.assertEqual(missing, [], f"app.js 引用了 index.html 里不存在的元素：{missing}")

    def test_runs_api_returns_filtered_pages_and_total(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        ids = [f"pagination-{uuid}" for uuid in ("a", "b", "c")]
        with server.db() as connection:
            for index, run_id in enumerate(ids):
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,passed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "load", "T0", "openai", "https://pagination.example.test", "m", "completed", 0,
                     f"{now[:-6]}{index:02d}+00:00", int(index != 1)),
                )
        try:
            first = self.get_json("/api/runs?page=1&page_size=2&conclusion=pass")
            second = self.get_json("/api/runs?page=2&page_size=2&conclusion=pass")
            searched = self.get_json("/api/runs?page=1&page_size=2&q=pagination.example.test")
            self.assertEqual((first["total"], len(first["items"])), (2, 2))
            self.assertEqual((second["total"], len(second["items"])), (2, 0))
            self.assertEqual(searched["total"], 3)
        finally:
            with server.db() as connection:
                connection.executemany("DELETE FROM runs WHERE id=?", [(run_id,) for run_id in ids])

    def test_delete_run_removes_record_and_result_directory(self):
        run_id = f"delete-test-{time.time_ns()}"
        result_dir = server.RESULTS / run_id
        result_dir.mkdir(parents=True)
        (result_dir / "report.md").write_text("test report", encoding="utf-8")
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, "load", "T0", "openai", "https://example.com", "m", "completed", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), str(result_dir.relative_to(server.ROOT))),
            )
        request = Request(self.base + f"/api/runs/{run_id}", method="DELETE")
        with urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["deleted_records"], 1)
        self.assertFalse(result_dir.exists())
        with server.db() as connection:
            self.assertIsNone(connection.execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone())

    def test_create_run_does_not_persist_key(self):
        payload = json.dumps({"protocol":"anthropic", "test_kind":"load", "base_url":"https://example.com", "api_key":"super-secret", "model":"ep-test", "case":"T0", "quick":True}).encode()
        request = Request(self.base + "/api/runs", data=payload, method="POST", headers={"Content-Type":"application/json"})
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(request) as response: created=json.load(response)
            self.assertEqual(created["status"], "queued")
            self.assertNotIn("api_key", created)
            self.assertEqual(created["protocol"], "anthropic")
            self.assertEqual(created["test_kind"], "load")
            self.assertEqual(put.call_args.args[0]["api_key"], "super-secret")
        self.assertNotIn("super-secret", self.db_path.read_bytes().decode("utf-8", "ignore"))

    def test_create_openai_responses_load_run(self):
        payload=json.dumps({"protocol":"openai_responses", "test_kind":"load", "base_url":"https://mux.la", "api_key":"secret", "model":"gpt-5-mini", "case":"T0", "quick":True}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["protocol"], "openai_responses")
        self.assertEqual(result["base_url"], "https://mux.la")
        self.assertEqual(put.call_args.args[0]["protocol"], "openai_responses")

    def test_create_load_presets(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"SIMPLE_LOAD", "quick":True}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["case_ids"], ["SIMPLE_LOAD"])
        self.assertTrue(result["is_batch"])
        children=put.call_args.args[0]["children"]
        self.assertEqual([child["case"] for child in children], ["L1","L2","L3","L4","L5","O1"])
        self.assertEqual([(child["requests"],child["concurrency"]) for child in children], [(50,30)]*5+[(100,10)])
        self.assertTrue(all(child["warmup_seconds"] == 0 for child in children))
        self.assertTrue(all(not child["quick"] for child in children))

        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"FULL_LOAD"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})):
                pass
        children=put.call_args.args[0]["children"]
        self.assertEqual([(child["requests"],child["concurrency"]) for child in children], [(200,50)]*5+[(100,10)])

    def test_create_random_load_uses_exact_manual_request_and_concurrency_values(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"RANDOM_LOAD", "quick":True, "requests":37, "concurrency":6}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        job=put.call_args.args[0]
        self.assertEqual(result["case_ids"], ["RANDOM_LOAD"])
        self.assertEqual((job["requests"],job["concurrency"]),(37,6))
        self.assertFalse(job["quick"])

    def test_reject_removed_strict_suite(self):
        payload=json.dumps({"protocol":"anthropic", "test_kind":"strict", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"STRICT_T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_create_kvv_run(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"kvv", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"KVV_K3_FEATURES"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "kvv")
        self.assertEqual(put.call_args.args[0]["case"], "KVV_K3_FEATURES")

    def test_official_k3_entries_are_visible_and_runnable_by_default(self):
        with urlopen(self.base + "/api/cases") as response:
            entries = {item["id"]: item for item in json.load(response)}
        for name, count in (("KVV_K3_DYNAMIC_TOOLS", 54), ("KVV_K3_THINKING_EFFORT", 26), ("KVV_PROMPT_TOKENS", 55)):
            with self.subTest(case=name):
                self.assertEqual(entries[name]["test_kind"], "kvv")
                self.assertEqual(entries[name]["requests"], count)
                payload = json.dumps({"protocol": "openai", "test_kind": "kvv", "base_url": "https://example.com", "api_key": "secret", "model": "kimi-k3", "case": name}).encode()
                with patch.object(server.JOB_QUEUE, "put") as put:
                    with urlopen(Request(self.base + "/api/runs", data=payload, method="POST", headers={"Content-Type": "application/json"})) as response:
                        result = json.load(response)
                self.assertEqual(result["test_kind"], "kvv")
                self.assertEqual(put.call_args.args[0]["case"], name)

    def test_gpt_image_list_has_one_load_entry_with_all_models(self):
        entries = [item for item in self.get_json("/api/cases") if item["test_kind"] == "gpt-image"]
        self.assertEqual([item["id"] for item in entries], ["GPT_IMAGE_2", "GPT_IMAGE_4K_TOKEN_VERIFY"])
        self.assertEqual(entries[0]["title"], "gpt-image-2 / 2.5")
        self.assertEqual(entries[0]["image_models"], ["gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"])
        self.assertIn("max", entries[0]["image_qualities"])

    def test_merged_image_run_validates_model_quality_combinations(self):
        payload = {
            "protocol": "openai_images", "test_kind": "gpt-image",
            "base_url": "https://example.com", "api_key": "secret",
            "model": "gpt-image-2", "case": "GPT_IMAGE_2",
            "requests": 7, "concurrency": 3,
            "image_models": ["gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"],
            "image_quality": ["high", "max"],
        }
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(payload).encode())
        self.assertEqual(status, 202)
        job = put.call_args.args[0]
        self.assertEqual(job["image_models"], payload["image_models"])
        self.assertEqual(job["image_quality"], ["high", "max"])
        self.assertEqual(job["requests"], 7)
        self.assertFalse(job.get("is_batch"))
        payload["image_quality"] = ["max"]
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, body = self._post_run(json.dumps(payload).encode())
        self.assertEqual(status, 400)
        self.assertIn("gpt-image-2 不支持所选 quality", body["error"])
        put.assert_not_called()
        payload["image_models"] = ["gpt-image-2.5-flare"]
        with patch.object(server.JOB_QUEUE, "put"):
            status, _ = self._post_run(json.dumps(payload).encode())
        self.assertEqual(status, 202)

    def test_create_gpt_image_run_keeps_randomization_choices(self):
        payload=json.dumps({
            "protocol":"openai_images",
            "test_kind":"gpt-image",
            "base_url":"https://api.openai.com",
            "api_key":"secret",
            "model":"gpt-image-2",
            "case":"GPT_IMAGE_2",
            "requests":7,
            "concurrency":3,
            "image_quality":["low","high"],
            "image_models":["gpt-image-2"],
            "image_sizes":["1024x1024","1536x1024"],
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "gpt-image")
        job=put.call_args.args[0]
        self.assertEqual((job["requests"],job["concurrency"]),(7,3))
        self.assertEqual(job["image_quality"],["low","high"])
        self.assertEqual(job["image_sizes"],["1024x1024","1536x1024"])

    def test_image_prompt_is_validated_and_passed_to_every_image_job(self):
        prompt = '  橘猫，晨光。\n保留 "引号"。  '
        for kind, protocol, cases, model in (
            ("gpt-image", "openai_images", ["GPT_IMAGE_2"], "gpt-image-2"),
            ("gemini-image", "gemini_images", ["GEMINI_IMAGE_GEN"], "gemini-3-pro-image"),
        ):
            payload = {"test_kind": kind, "protocol": protocol, "cases": cases, "model": model,
                       "base_url": "https://example.com", "api_key": "secret", "requests": 2, "concurrency": 1}
            for value, expected in ((prompt, prompt), (None, None), (" \n ", None), ("猫" * 8000, "猫" * 8000)):
                with self.subTest(kind=kind, cases=cases, prompt_length=len(value or "")):
                    with patch.object(server.JOB_QUEUE, "put") as put:
                        status, _ = self._post_run(json.dumps(dict(payload, image_prompt=value)).encode())
                    self.assertEqual(status, 202)
                    job = put.call_args.args[0]
                    jobs = job["children"] if job.get("is_batch") else [job]
                    self.assertTrue(all(child["image_prompt"] == expected for child in jobs))
            for invalid in ("猫" * 8001, 123, False, ["cat"], "cat\x00dog"):
                with patch.object(server.JOB_QUEUE, "put"):
                    status, body = self._post_run(json.dumps(dict(payload, image_prompt=invalid)).encode())
                self.assertEqual(status, 400)
                self.assertIn("提示词", body["error"])
        fixed = dict(payload, test_kind="gpt-image", protocol="openai_images", model="gpt-image-2",
                     cases=["GPT_IMAGE_4K_TOKEN_VERIFY"], image_prompt=prompt)
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(fixed).encode())
        self.assertEqual(status, 202)
        self.assertIsNone(put.call_args.args[0]["image_prompt"])

    def test_image_prompt_worker_preserves_text_and_clears_stale_environment(self):
        prompt = '  中文\n"quoted" $(literal)  '
        for kind, protocol, case, variable in (
            ("gpt-image", "openai_images", "GPT_IMAGE_2", "GPT_IMAGE_PROMPT"),
            ("gemini-image", "gemini_images", "GEMINI_IMAGE_GEN", "GEMINI_IMAGE_PROMPT"),
        ):
            job = {"test_kind": kind, "protocol": protocol, "case": case, "requests": 2, "concurrency": 1}
            with patch.dict(os.environ, {variable: "旧提示词"}):
                self.assertEqual(self._execute_child_capture(job)["env"][variable], "")
                captured = self._execute_child_capture(dict(job, image_prompt=prompt))
            self.assertEqual(captured["env"][variable], prompt)
            self.assertNotIn(prompt, captured["command"])

    def test_create_fixed_gpt_image_4k_token_verification(self):
        payload=json.dumps({
            "protocol":"openai_images",
            "test_kind":"gpt-image",
            "base_url":"https://api.openai.com",
            "api_key":"secret",
            "model":"gpt-image-2",
            "case":"GPT_IMAGE_4K_TOKEN_VERIFY",
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "gpt-image")
        job=put.call_args.args[0]
        self.assertEqual((job["requests"],job["concurrency"]),(3,1))
        self.assertEqual(job["image_quality"],["high","max"])
        self.assertEqual(job["image_models"],["gpt-image-2","gpt-image-2.5-flare","gpt-image-2.5-sunburst"])
        self.assertEqual(job["image_sizes"],["3840x2160"])

    def test_gpt_image_load_options_validation_and_defaults(self):
        payload = {
            "protocol": "openai_images", "test_kind": "gpt-image",
            "base_url": "https://example.com", "api_key": "secret",
            "model": "gpt-image-2", "case": "GPT_IMAGE_2",
            "requests": 2, "concurrency": 1, "image_sizes": ["1K", "2K", "4K"],
        }
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(payload).encode())
        self.assertEqual(status, 202)
        self.assertIsNone(put.call_args.args[0]["image_aspect_ratio"])
        self.assertEqual(put.call_args.args[0]["image_request_timeout_seconds"], 300)
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(dict(payload, image_aspect_ratio="21:9", image_request_timeout_seconds=12.5)).encode())
        self.assertEqual(status, 202)
        self.assertEqual(put.call_args.args[0]["image_aspect_ratio"], "21:9")
        self.assertEqual(put.call_args.args[0]["image_request_timeout_seconds"], 12.5)
        for invalid in (0, -1, True, "bad", "nan", "inf"):
            with self.subTest(timeout=invalid), patch.object(server.JOB_QUEUE, "put") as put:
                status, _ = self._post_run(json.dumps(dict(payload, image_request_timeout_seconds=invalid)).encode())
                self.assertEqual(status, 400)
                put.assert_not_called()
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(dict(payload, image_aspect_ratio="7:3")).encode())
        self.assertEqual(status, 400)
        put.assert_not_called()
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(dict(payload, case="GPT_IMAGE_4K_TOKEN_VERIFY", image_aspect_ratio="21:9", image_request_timeout_seconds=12.5)).encode())
        self.assertEqual(status, 202)
        self.assertIsNone(put.call_args.args[0]["image_aspect_ratio"])
        self.assertIsNone(put.call_args.args[0]["image_request_timeout_seconds"])

    def test_gpt_image_worker_options_override_stale_environment(self):
        job = {"test_kind": "gpt-image", "case": "GPT_IMAGE_2", "protocol": "openai_images", "requests": 2, "concurrency": 1}
        with patch.dict(os.environ, {"GPT_IMAGE_ASPECT_RATIO": "16:9"}):
            env = self._execute_child_capture(job)["env"]
        self.assertEqual(env["GPT_IMAGE_ASPECT_RATIO"], "")
        env = self._execute_child_capture(dict(job, image_aspect_ratio="3:4", image_request_timeout_seconds=12.5))["env"]
        self.assertEqual(env["GPT_IMAGE_ASPECT_RATIO"], "3:4")
        self.assertEqual(env["GPT_IMAGE_REQUEST_TIMEOUT_SECONDS"], "12.5")

    def test_gpt_image_hold_seconds_is_optional_and_validated(self):
        payload = {
            "protocol": "openai_images", "test_kind": "gpt-image",
            "base_url": "https://example.com", "api_key": "secret",
            "model": "gpt-image-2", "case": "GPT_IMAGE_2",
            "requests": 100, "concurrency": 2,
        }
        # 默认不启用：缺键 / 显式 null / 空串一律按「不启用」入库。
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(payload).encode())
        self.assertEqual(status, 202)
        self.assertIsNone(put.call_args.args[0]["image_hold_seconds"])
        for disabled in (None, ""):
            with self.subTest(disabled=disabled), patch.object(server.JOB_QUEUE, "put") as put:
                status, _ = self._post_run(json.dumps(dict(payload, image_hold_seconds=disabled)).encode())
                self.assertEqual(status, 202)
                self.assertIsNone(put.call_args.args[0]["image_hold_seconds"])
        # 勾选启用后与总请求数上限同时生效。
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(dict(payload, image_hold_seconds=30.5)).encode())
        self.assertEqual(status, 202)
        job = put.call_args.args[0]
        self.assertEqual(job["image_hold_seconds"], 30.5)
        self.assertEqual((job["requests"], job["concurrency"]), (100, 2))
        for invalid in (0, -1, True, "bad", "nan", "inf"):
            with self.subTest(invalid=invalid), patch.object(server.JOB_QUEUE, "put") as put:
                status, _ = self._post_run(json.dumps(dict(payload, image_hold_seconds=invalid)).encode())
                self.assertEqual(status, 400)
                put.assert_not_called()
        # 4K Token 验真是固定 3 条请求的验真批，不接受压测持续时长。
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(json.dumps(dict(payload, case="GPT_IMAGE_4K_TOKEN_VERIFY", image_hold_seconds=30.5)).encode())
        self.assertEqual(status, 202)
        self.assertIsNone(put.call_args.args[0]["image_hold_seconds"])

    def test_gpt_image_worker_clears_stale_hold_environment(self):
        job = {"test_kind": "gpt-image", "case": "GPT_IMAGE_2", "protocol": "openai_images", "requests": 100, "concurrency": 2}
        with patch.dict(os.environ, {"GPT_IMAGE_HOLD_SECONDS": "999"}):
            env = self._execute_child_capture(job)["env"]
        # 不启用时必须写空串：否则父进程残留会把单批被动变成时长压测。
        self.assertEqual(env["GPT_IMAGE_HOLD_SECONDS"], "")
        env = self._execute_child_capture(dict(job, image_hold_seconds=45.5))["env"]
        self.assertEqual(env["GPT_IMAGE_HOLD_SECONDS"], "45.5")

    def test_create_gemini_image_run_keeps_randomization_choices(self):
        payload=json.dumps({
            "protocol":"gemini_images",
            "test_kind":"gemini-image",
            "base_url":"https://generativelanguage.googleapis.com",
            "api_key":"secret",
            "model":"gemini-3-pro-image",
            "case":"GEMINI_IMAGE_GEN",
            "requests":6,
            "concurrency":2,
            "gemini_quality":["1K","4K"],
            "gemini_models":["gemini-3-pro-image"],
            "gemini_aspects":["1:1","16:9"],
            "gemini_counts":[1,2],
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "gemini-image")
        job=put.call_args.args[0]
        self.assertEqual((job["requests"],job["concurrency"]),(6,2))
        self.assertEqual(job["gemini_quality"],["1K","4K"])
        self.assertEqual(job["gemini_models"],["gemini-3-pro-image"])
        self.assertEqual(job["gemini_aspects"],["1:1","16:9"])
        self.assertEqual(job["gemini_counts"],[1,2])

    # —— Seedance 视频生成（异步任务面） ——
    def _seedance_payload(self, **overrides):
        body = {
            "protocol": "seedance", "test_kind": "seedance",
            "base_url": "https://token-hub.trankeji.example", "api_key": "secret",
            "model": "doubao-seedance-1-0-pro-250528", "case": "SEEDANCE_VIDEO_GEN",
            "requests": 2, "concurrency": 1, "seedance_prompt": "一只白鹭在浅滩展翅",
        }
        body.update(overrides)
        return json.dumps(body).encode()

    def _post_run(self, payload):
        """提交一次任务并返回 (HTTP 状态, 解析后的 JSON)。

        入队成功是 202（Accepted），不是 200；合法提交一律断言 202，用 200 会误判成失败。
        """
        try:
            with urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def test_create_seedance_run_keeps_explicit_video_parameters(self):
        payload = self._seedance_payload(
            seedance_resolution="1080p", seedance_ratio="9:16", seedance_duration=12,
            seedance_image_role="reference_image", seedance_image_url="https://cdn.example/first.png",
            seedance_audio=True, seedance_watermark=True,
        )
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, result = self._post_run(payload)
        self.assertEqual(status, 202)
        self.assertEqual(result["test_kind"], "seedance")
        job = put.call_args.args[0]
        self.assertEqual((job["requests"], job["concurrency"]), (2, 1))
        self.assertEqual(job["seedance_resolution"], "1080p")
        self.assertEqual(job["seedance_ratio"], "9:16")
        self.assertEqual(job["seedance_duration"], 12)
        self.assertEqual(job["seedance_image_role"], "reference_image")
        self.assertEqual(job["seedance_image_url"], "https://cdn.example/first.png")
        self.assertTrue(job["seedance_audio"])
        self.assertTrue(job["seedance_watermark"])
        self.assertEqual(job["seedance_prompt"], "一只白鹭在浅滩展翅")
        self.assertIsNone(job["seedance_image_path"])

    def test_seedance_case_payload_exposes_the_generation_modes(self):
        """生成方式决定面板显隐与请求体形状，取值域同样由后端下发（前端不写死）。"""
        payload = next(x for x in self.get_json("/api/cases") if x["id"] == "SEEDANCE_VIDEO_GEN")
        self.assertEqual([item["id"] for item in payload["seedance_modes"]],
                         ["text", "image", "video", "both"])
        self.assertEqual(payload["seedance_default_mode"], "text")
        self.assertIn("仅文生视频", payload["seedance_modes"][0]["label"])
        # 提示文案必须写清"地址才算提交"，否则用户会以为上传了本地文件就等于带上了参考视频。
        self.assertIn("视频地址", payload["seedance_mode_note"])

    def test_seedance_run_enforces_the_generation_mode(self):
        # 显式声明生成方式时与输入严格对齐：少给（声明带图却没有图）和多给（声明文生视频却
        # 带着图）都在入队时拦回——该网关把参数校验放在异步执行阶段，错配的代价是一次完整生成。
        for overrides, marker in (
            ({"seedance_mode": "text", "seedance_image_url": "https://cdn.example/a.png"}, "不会提交图片"),
            ({"seedance_mode": "text", "seedance_video_url": "https://cdn.example/a.mp4"}, "不会提交参考视频"),
            ({"seedance_mode": "image"}, "但没有图片"),
            ({"seedance_mode": "video"}, "但没有视频地址"),
            ({"seedance_mode": "both", "seedance_image_url": "https://cdn.example/a.png"}, "但没有视频地址"),
            ({"seedance_mode": "image", "seedance_image_url": "https://cdn.example/a.png",
              "seedance_video_url": "https://cdn.example/a.mp4"}, "不会提交参考视频"),
            ({"seedance_mode": "video2video", "seedance_video_url": "https://cdn.example/a.mp4"}, "生成方式"),
        ):
            with self.subTest(overrides=overrides), patch.object(server.JOB_QUEUE, "put"):
                status, body = self._post_run(self._seedance_payload(**overrides))
            self.assertEqual(status, 400)
            self.assertIn(marker, body["error"])

        # 未声明生成方式（直接打 API 的调用方）按素材推导，不做冲突校验——老调用方式不受影响。
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(self._seedance_payload(seedance_image_url="https://cdn.example/a.png"))
        self.assertEqual(status, 202)
        self.assertEqual(put.call_args.args[0]["seedance_mode"], "image")

    def test_create_seedance_run_keeps_the_generation_mode(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(self._seedance_payload(
                seedance_mode="both", seedance_image_url="https://cdn.example/a.png",
                seedance_video_url="https://cdn.example/a.mp4",
            ))
        self.assertEqual(status, 202)
        job = put.call_args.args[0]
        self.assertEqual(job["seedance_mode"], "both")
        self.assertEqual(job["seedance_image_url"], "https://cdn.example/a.png")
        self.assertEqual(job["seedance_video_url"], "https://cdn.example/a.mp4")

    def test_seedance_case_payload_exposes_the_reference_video_domains(self):
        """参考视频的取值域同样由后端下发（前端不写死角色/体积上限）。"""
        payload = next(x for x in self.get_json("/api/cases") if x["id"] == "SEEDANCE_VIDEO_GEN")
        roles = [item["id"] for item in payload["seedance_video_roles"]]
        self.assertIn("reference_video", roles)
        # 空串是合法取值（= 不带 role 字段），必须留在下拉里。
        self.assertIn("", roles)
        self.assertEqual(payload["seedance_default_video_role"], "reference_video")
        self.assertEqual(payload["seedance_video_max_bytes"], 64 * 1024 * 1024)
        self.assertIn("video/mp4", payload["seedance_video_mime_types"])
        self.assertIn("mp4", payload["seedance_video_suffixes"])
        self.assertIn("只接受 URL", payload["seedance_video_note"])

    def test_create_seedance_run_applies_documented_defaults(self):
        # 分辨率 / 比例 / 时长 / 图片位置都可以不填，缺省值由后端补（不在前端写死）。
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(self._seedance_payload())
        self.assertEqual(status, 202)
        job = put.call_args.args[0]
        self.assertEqual(job["seedance_resolution"], "720p")
        self.assertEqual(job["seedance_ratio"], "adaptive")
        self.assertEqual(job["seedance_duration"], 5)
        self.assertEqual(job["seedance_image_role"], "first_frame")
        # 官方默认：generate_audio 开（有声视频）、watermark 关；两个开关都会显式提交布尔值。
        self.assertTrue(job["seedance_audio"])
        self.assertFalse(job["seedance_watermark"])

    def test_seedance_run_validates_prompt_and_value_domains(self):
        # 提示词必填：视频任务没有内容就没有可执行的用例。
        with patch.object(server.JOB_QUEUE, "put"):
            status, body = self._post_run(self._seedance_payload(seedance_prompt="   "))
        self.assertEqual(status, 400)
        self.assertIn("提示词", body["error"])
        with patch.object(server.JOB_QUEUE, "put"):
            status, body = self._post_run(self._seedance_payload(seedance_resolution="8k"))
        self.assertEqual(status, 400)
        self.assertIn("分辨率", body["error"])
        with patch.object(server.JOB_QUEUE, "put"):
            status, body = self._post_run(self._seedance_payload(seedance_ratio="2:1"))
        self.assertEqual(status, 400)
        self.assertIn("画面比例", body["error"])
        with patch.object(server.JOB_QUEUE, "put"):
            status, body = self._post_run(self._seedance_payload(seedance_image_role="middle_frame"))
        self.assertEqual(status, 400)
        self.assertIn("图片位置", body["error"])

    def test_seedance_run_rejects_duration_requests_and_concurrency_out_of_range(self):
        # requests 的通用上限是 100000，seedance 自己的上限是 100 —— 这里要钉的是**套件级**
        # 上限真的生效（提交量大等于直接把钱烧掉），所以只测 seedance 私有区间。
        # 注意默认模型（doubao-seedance-1-0-pro-250528）识别不出 2.x 档位，时长区间退化成
        # 并集 4–30，因此 16 秒这一档不在这里判——按档位收紧的用例见下一个循环。
        for overrides, marker in (
            ({"seedance_duration": 3}, "时长"),
            ({"seedance_duration": 31}, "时长"),
            ({"seedance_duration": 5.5}, "时长"),
            ({"requests": 101}, "视频任务数"),
            ({"concurrency": 17}, "并发数"),
        ):
            with self.subTest(overrides=overrides), patch.object(server.JOB_QUEUE, "put"):
                status, body = self._post_run(self._seedance_payload(**overrides))
            self.assertEqual(status, 400)
            self.assertIn(marker, body["error"])

    def test_seedance_run_enforces_duration_by_model_tier(self):
        """时长上限按**模型档位**解析：2.0 → 15，2.5 → 30。

        该网关把参数校验放在异步执行阶段，档位越界不会在提交期报错（首跑就是拿 480p 打
        2.0，整条任务跑完 4.6 分钟才失败），所以必须在入队时按档位拦下来。
        """
        for model, duration, expected in (
            ("Seedance-2.0", 15, 202),
            ("Seedance-2.0", 16, 400),
            ("Seedance-2.5", 16, 202),
            ("Seedance-2.5", 30, 202),
            ("Seedance-2.5", 31, 400),
        ):
            with self.subTest(model=model, duration=duration), patch.object(server.JOB_QUEUE, "put"):
                status, body = self._post_run(
                    self._seedance_payload(model=model, seedance_duration=duration)
                )
            self.assertEqual(status, expected, body)
            if expected == 400:
                self.assertIn("时长", body["error"])

    def test_seedance_run_requires_manual_scale_and_single_case(self):
        body = json.loads(self._seedance_payload())
        body.pop("requests"); body.pop("concurrency")
        with patch.object(server.JOB_QUEUE, "put"):
            status, error = self._post_run(json.dumps(body).encode())
        self.assertEqual(status, 400)
        self.assertIn("视频任务数", error["error"])
        # cases 会先被去重，所以要用两个**不同**的 id 才能触发「一次只能选一个」。
        with patch.object(server.JOB_QUEUE, "put"):
            status, error = self._post_run(self._seedance_payload(
                cases=["SEEDANCE_VIDEO_GEN", "SEEDANCE_VIDEO_GEN_ALT"]))
        self.assertEqual(status, 400)
        self.assertIn("只能选择一个", error["error"])

    def test_seedance_upload_path_must_live_in_the_upload_directory(self):
        """上传图片只认本服务上传目录里的文件 —— 这是一条任意文件读取的防线。

        请求体里的 seedance_image_path 是不可信输入：必须解析后落在 web/data/uploads
        之内，否则任何绝对路径都能把服务器上的文件塞进请求。
        """
        with patch.object(server.JOB_QUEUE, "put"):
            status, body = self._post_run(self._seedance_payload(seedance_image_path="/etc/hosts"))
        self.assertEqual(status, 400)
        self.assertIn("图片", body["error"])
        # 目录存在但在上传目录之外的仓库内文件同样要拦（相对路径穿越）。
        with patch.object(server.JOB_QUEUE, "put"):
            status, _ = self._post_run(self._seedance_payload(seedance_image_path="README.md"))
        self.assertEqual(status, 400)

    def test_seedance_run_accepts_an_uploaded_image_and_persists_relative_path(self):
        upload_dir = Path(server.SEEDANCE_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded = upload_dir / f"test-{time.time_ns()}.png"
        uploaded.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        try:
            relative = str(uploaded.resolve().relative_to(Path(server.ROOT)))
            with patch.object(server.JOB_QUEUE, "put") as put:
                status, _ = self._post_run(self._seedance_payload(seedance_image_path=relative))
            self.assertEqual(status, 202)
            self.assertEqual(put.call_args.args[0]["seedance_image_path"], relative)
        finally:
            uploaded.unlink(missing_ok=True)

    def test_seedance_upload_endpoint_accepts_images_and_rejects_other_types(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        request = Request(
            self.base + "/api/uploads", data=png, method="POST",
            headers={"Content-Type": "image/png", "X-File-Name": "first%E5%B8%A7.png"},
        )
        with urlopen(request) as response:
            result = json.load(response)
        saved = Path(server.ROOT) / result["path"]
        try:
            self.assertTrue(saved.is_file(), "上传的图片没有落到上传目录")
            self.assertEqual(saved.read_bytes(), png)
            # 落盘名由服务端随机生成（uuid4 hex + 由服务端判定的扩展名），不采信客户端文件名，
            # 客户端只拿到回显用的展示名——否则文件名就是一条路径注入面。
            self.assertEqual(saved.parent.resolve(), Path(server.SEEDANCE_UPLOAD_DIR).resolve())
            self.assertRegex(saved.name, r"^[0-9a-f]{32}\.png$")
            self.assertEqual(result["name"], "first%E5%B8%A7.png")
            self.assertRegex(result["path"], r"^web/data/uploads/[0-9a-f]{32}\.png$")
            self.assertEqual(result["mime"], "image/png")
            self.assertEqual(result["bytes"], len(png))
            self.assertIn("swept", result)
        finally:
            saved.unlink(missing_ok=True)

        for content_type in ("text/plain", "application/json", "image/svg+xml"):
            with self.subTest(content_type=content_type), self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/api/uploads", data=b"x", method="POST",
                                headers={"Content-Type": content_type}))
            self.assertEqual(caught.exception.code, 400)

    def test_seedance_upload_endpoint_accepts_videos_with_a_larger_cap(self):
        """参考视频走同一条原始 body 通道，但用**自己**的体积上限（64 MiB）。"""
        import seedance.cases as seedance_cases
        clip = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        request = Request(
            self.base + "/api/uploads", data=clip, method="POST",
            headers={"Content-Type": "video/mp4", "X-File-Name": "ref.mp4"},
        )
        with urlopen(request) as response:
            result = json.load(response)
        saved = Path(server.ROOT) / result["path"]
        try:
            self.assertEqual(result["kind"], "video")
            self.assertEqual(result["mime"], "video/mp4")
            self.assertRegex(saved.name, r"^[0-9a-f]{32}\.mp4$")
            self.assertEqual(saved.read_bytes(), clip)
        finally:
            saved.unlink(missing_ok=True)

        # 视频上限与图片上限互不牵连：把图片上限压到 8 字节，16 字节的视频仍要能过。
        with patch.object(seedance_cases, "SEEDANCE_MAX_IMAGE_BYTES", 8):
            with urlopen(Request(self.base + "/api/uploads", data=b"\x00" * 16, method="POST",
                                 headers={"Content-Type": "video/webm"})) as response:
                allowed = json.load(response)
        (Path(server.ROOT) / allowed["path"]).unlink(missing_ok=True)
        self.assertEqual(allowed["kind"], "video")

        # 反过来，视频上限生效时超限必须 400。
        with patch.object(seedance_cases, "SEEDANCE_MAX_INPUT_VIDEO_BYTES", 8):
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/api/uploads", data=b"\x00" * 16, method="POST",
                                headers={"Content-Type": "video/mp4"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("过大", json.loads(caught.exception.read())["error"])

        # 非白名单的视频容器仍要拒（例如 .avi）。
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/uploads", data=b"x", method="POST",
                            headers={"Content-Type": "video/x-msvideo"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("video/mp4", json.loads(caught.exception.read())["error"])

    def test_seedance_run_accepts_uploaded_video_and_validates_its_fields(self):
        upload_dir = Path(server.SEEDANCE_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded = upload_dir / f"test-{time.time_ns()}.mp4"
        uploaded.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 32)
        try:
            relative = str(uploaded.resolve().relative_to(Path(server.ROOT)))
            with patch.object(server.JOB_QUEUE, "put") as put:
                status, _ = self._post_run(self._seedance_payload(
                    seedance_video_path=relative,
                    seedance_video_url="https://cdn.example/ref.mp4",
                    seedance_video_role="",
                ))
            self.assertEqual(status, 202)
            job = put.call_args.args[0]
            self.assertEqual(job["seedance_video_path"], relative)
            self.assertEqual(job["seedance_video_url"], "https://cdn.example/ref.mp4")
            # 空串是合法取值：表示"不带 role 字段"，不能被当成缺省值改写。
            self.assertEqual(job["seedance_video_role"], "")
        finally:
            uploaded.unlink(missing_ok=True)

        # 缺省角色由后端补（前端不写死）。
        with patch.object(server.JOB_QUEUE, "put") as put:
            status, _ = self._post_run(self._seedance_payload())
        self.assertEqual(status, 202)
        self.assertEqual(put.call_args.args[0]["seedance_video_role"], "reference_video")
        self.assertIsNone(put.call_args.args[0]["seedance_video_path"])
        # URL 字段沿用它自己的约定：未填是空串（与 seedance_image_url 一致）。
        self.assertEqual(put.call_args.args[0]["seedance_video_url"], "")

        for overrides, marker in (
            ({"seedance_video_role": "first_frame"}, "参考视频位置"),
            ({"seedance_video_url": "ftp://cdn.example/ref.mp4"}, "视频地址"),
            ({"seedance_video_path": "/etc/hosts"}, "参考视频"),
            ({"seedance_video_path": "README.md"}, "参考视频"),
        ):
            with self.subTest(overrides=overrides), patch.object(server.JOB_QUEUE, "put"):
                status, body = self._post_run(self._seedance_payload(**overrides))
            self.assertEqual(status, 400)
            self.assertIn(marker, body["error"])

    def test_seedance_uploaded_video_suffix_must_be_a_video_container(self):
        """上传目录内的文件也要看后缀：把 .png 当参考视频传进来必须拦下。"""
        upload_dir = Path(server.SEEDANCE_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        disguised = upload_dir / f"test-{time.time_ns()}.png"
        disguised.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        try:
            relative = str(disguised.resolve().relative_to(Path(server.ROOT)))
            with patch.object(server.JOB_QUEUE, "put"):
                status, body = self._post_run(self._seedance_payload(seedance_video_path=relative))
            self.assertEqual(status, 400)
            self.assertIn("格式", body["error"])
        finally:
            disguised.unlink(missing_ok=True)

    def test_release_seedance_upload_removes_image_and_video(self):
        """临时素材回收必须覆盖两类文件，否则 web/data/uploads 会无上限增长。"""
        upload_dir = Path(server.SEEDANCE_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        image = upload_dir / f"test-{time.time_ns()}.png"
        video = upload_dir / f"test-{time.time_ns()}.mp4"
        image.write_bytes(b"i"); video.write_bytes(b"v")
        relative = lambda path: str(path.resolve().relative_to(Path(server.ROOT)))
        server._release_seedance_upload({
            "seedance_image_path": relative(image), "seedance_video_path": relative(video),
        })
        self.assertFalse(image.exists())
        self.assertFalse(video.exists())

    def test_seedance_upload_endpoint_enforces_the_size_cap(self):
        import seedance.cases as seedance_cases
        oversized = Request(
            self.base + "/api/uploads", data=b"\x00" * 16, method="POST",
            headers={"Content-Type": "image/png"},
        )
        with patch.object(seedance_cases, "SEEDANCE_MAX_IMAGE_BYTES", 8):
            with self.assertRaises(HTTPError) as caught:
                urlopen(oversized)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("过大", json.loads(caught.exception.read())["error"])

    def test_seedance_suite_rejects_other_protocols(self):
        # 视频任务面不能借文字/图像协议的线格式发出去：协议白名单必须真的挡在入口。
        for protocol in ("openai", "volcengine", "anthropic", "gemini", "openai_images", "gemini_images"):
            with self.subTest(protocol=protocol), patch.object(server.JOB_QUEUE, "put"):
                status, _ = self._post_run(self._seedance_payload(protocol=protocol))
            self.assertEqual(status, 400)

    def test_seedance_child_dispatches_to_the_video_runner_with_its_params(self):
        """test_kind→脚本的派发链最容易漏（漏了就是「提交成功但立刻失败」）。

        视频内容参数走环境变量（提示词含换行/引号；图片给的是本机路径、由执行器读成
        data URI 再发出去），执行规模（任务数 / 并发）走命令行——两者都必须真的到达子进程。
        """
        captured = self._execute_child_capture({
            "test_kind": "seedance", "case": "SEEDANCE_VIDEO_GEN", "protocol": "seedance",
            "base_url": "https://token-hub.trankeji.example", "model": "doubao-seedance-1-0-pro-250528",
            "requests": 3, "concurrency": 2,
            "seedance_prompt": "一只白鹭在浅滩展翅\n第二行：镜头缓慢推进",
            "seedance_resolution": "1080p", "seedance_ratio": "9:16", "seedance_duration": 12,
            "seedance_mode": "both",
            "seedance_image_role": "reference_image",
            "seedance_image_path": "web/data/uploads/abc.png",
            "seedance_image_url": "https://cdn.example/last.png",
            "seedance_video_role": "reference_video",
            "seedance_video_path": "web/data/uploads/ref.mp4",
            "seedance_video_url": "https://cdn.example/ref.mp4",
            "seedance_audio": True, "seedance_watermark": True,
        })
        command = [str(part) for part in captured["command"]]
        self.assertEqual(Path(command[1]).name, "run_seedance.py")
        self.assertEqual(command[command.index("--requests") + 1], "3")
        self.assertEqual(command[command.index("--concurrency") + 1], "2")
        env = captured["env"]
        # api_root 决定版本前缀：seedance 走 /api/v3（不是 /v1）。
        self.assertEqual(env["SEEDANCE_BASE_URL"], "https://token-hub.trankeji.example/api/v3")
        self.assertEqual(env["SEEDANCE_API_KEY"], "sk-x")
        self.assertEqual(env["SEEDANCE_MODEL"], "doubao-seedance-1-0-pro-250528")
        self.assertEqual(env["SEEDANCE_PROMPT"], "一只白鹭在浅滩展翅\n第二行：镜头缓慢推进")
        self.assertEqual(env["SEEDANCE_RESOLUTION"], "1080p")
        self.assertEqual(env["SEEDANCE_RATIO"], "9:16")
        self.assertEqual(env["SEEDANCE_DURATION"], "12")
        self.assertEqual(env["SEEDANCE_MODE"], "both")
        self.assertEqual(env["SEEDANCE_IMAGE_ROLE"], "reference_image")
        self.assertEqual(env["SEEDANCE_IMAGE_PATH"], "web/data/uploads/abc.png")
        self.assertEqual(env["SEEDANCE_IMAGE_URL"], "https://cdn.example/last.png")
        self.assertEqual(env["SEEDANCE_VIDEO_ROLE"], "reference_video")
        self.assertEqual(env["SEEDANCE_VIDEO_PATH"], "web/data/uploads/ref.mp4")
        self.assertEqual(env["SEEDANCE_VIDEO_URL"], "https://cdn.example/ref.mp4")
        self.assertEqual(env["SEEDANCE_GENERATE_AUDIO"], "1")
        self.assertEqual(env["SEEDANCE_WATERMARK"], "1")

    def test_seedance_child_passes_cli_flags_that_the_runner_accepts(self):
        """CLI 旗标拼错会静默退化成默认值（甚至 argparse 直接报错退出），必须钉住。"""
        captured = self._execute_child_capture({
            "test_kind": "seedance", "case": "SEEDANCE_VIDEO_GEN", "protocol": "seedance",
            "requests": 1, "concurrency": 1,
        })
        env = captured["env"]
        self.assertEqual(env["SEEDANCE_GENERATE_AUDIO"], "0")
        self.assertEqual(env["SEEDANCE_WATERMARK"], "0")
        # 生成方式未给时是空串 = 执行器按素材推导（不是硬塞一个默认档，避免悄悄改行为）。
        self.assertEqual(env["SEEDANCE_MODE"], "")
        self.assertEqual(env["SEEDANCE_IMAGE_PATH"], "")
        self.assertEqual(env["SEEDANCE_VIDEO_PATH"], "")
        self.assertEqual(env["SEEDANCE_VIDEO_URL"], "")
        self.assertEqual([str(part) for part in captured["command"]][-4:],
                         ["--requests", "1", "--concurrency", "1"])
        # 执行器必须真的存在（脚本路径写错时 worker 只会报「未生成 summary」）。
        self.assertTrue((Path(server.ROOT) / "scripts" / "run_seedance.py").is_file())

    def test_sweep_stale_uploads_removes_only_old_files(self):
        upload_dir = Path(server.SEEDANCE_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        fresh = upload_dir / f"fresh-{time.time_ns()}.png"
        stale = upload_dir / f"stale-{time.time_ns()}.png"
        fresh.write_bytes(b"f"); stale.write_bytes(b"s")
        try:
            old = time.time() - server.SEEDANCE_UPLOAD_TTL_SECONDS - 60
            os.utime(stale, (old, old))
            removed = server.sweep_stale_uploads()
            self.assertGreaterEqual(removed, 1)
            self.assertTrue(fresh.is_file(), "新上传的图片不能被清理掉")
            self.assertFalse(stale.exists())
        finally:
            fresh.unlink(missing_ok=True)
            stale.unlink(missing_ok=True)

    def test_gemini_image_requires_manual_requests_and_concurrency(self):
        payload=json.dumps({
            "protocol":"gemini_images","test_kind":"gemini-image","base_url":"https://example.com",
            "api_key":"secret","model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
        }).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_gemini_image_rejects_unsupported_choices_and_protocol(self):
        bad_count=json.dumps({
            "protocol":"gemini_images","test_kind":"gemini-image","base_url":"https://example.com",
            "api_key":"secret","model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
            "requests":3,"concurrency":1,"gemini_counts":[9],
        }).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=bad_count,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

        bad_protocol=json.dumps({
            "protocol":"openai_images","test_kind":"gemini-image","base_url":"https://example.com",
            "api_key":"secret","model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
            "requests":3,"concurrency":1,
        }).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=bad_protocol,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_gemini_image_case_is_listed_with_its_parameter_pools(self):
        cases=self.get_json("/api/cases")
        gemini=[item for item in cases if item["test_kind"]=="gemini-image"]
        self.assertEqual([item["id"] for item in gemini],["GEMINI_IMAGE_GEN"])
        self.assertEqual(gemini[0]["image_qualities"],["1K","2K","4K"])
        self.assertEqual(gemini[0]["image_counts"],[1,2,3,4])
        self.assertIn("16:9", gemini[0]["image_aspect_ratios"])

    def test_gemini3_cases_are_listed_with_native_protocol(self):
        cases=self.get_json("/api/cases")
        gemini={item["id"]:item for item in cases if item["test_kind"]=="gemini"}
        # 「Gemini 测试」= 文本专项（原生 generateContent），与「Gemini 图片测试」是两个模块。
        self.assertEqual(set(gemini), {"GEMINI3_BASIC","GEMINI3_P0","GEMINI3_P1","GEMINI3_P2","GEMINI3_FINGERPRINT"})
        self.assertTrue(all(item["protocols"]==["gemini"] for item in gemini.values()))
        self.assertTrue(all(item["executor"]=="scripts/run_gemini3.py" for item in gemini.values()))
        self.assertEqual(gemini["GEMINI3_BASIC"]["requests"],11)
        self.assertEqual(gemini["GEMINI3_P0"]["requests"],28)
        self.assertEqual(gemini["GEMINI3_P1"]["requests"],13)
        self.assertEqual(gemini["GEMINI3_P2"]["requests"],17)
        self.assertEqual(gemini["GEMINI3_FINGERPRINT"]["requests"],16)
        # 两个模型共用一个档位组：前端只需要一级「模型」下拉。
        self.assertEqual({item["gemini_variant"] for item in gemini.values()}, {"gemini3"})
        self.assertEqual(gemini["GEMINI3_P0"]["default_model"],"gemini-3.8-flash")
        self.assertEqual([m["id"] for m in gemini["GEMINI3_P0"]["gemini_models"]],
                         ["gemini-3.8-flash","gemini-3.1-pro-preview"])
        self.assertTrue(all(m["available"] for m in gemini["GEMINI3_P0"]["gemini_models"]))
        # 判定口径文档必须随用例一起暴露，避免报告与文档脱节。
        self.assertIn("docs/Gemini3专项测试用例设计.md", gemini["GEMINI3_P0"]["admission"])
        self.assertIn("warning 与 blocked 仅记录", gemini["GEMINI3_P2"]["admission"])

    def test_create_gemini3_run_accepts_native_protocol(self):
        # 回归：/api/run 的 test_kind→用例集派发链若漏掉 gemini，任何提交都会 400「未知测试类型」。
        payload=json.dumps({"protocol":"gemini","test_kind":"gemini","base_url":"https://mux.la",
                            "api_key":"secret","model":"gemini-3.8-flash",
                            "cases":["GEMINI3_BASIC"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"],"gemini")
        job=put.call_args.args[0]
        self.assertEqual(job["case"],"GEMINI3_BASIC")
        self.assertEqual(job["protocol"],"gemini")
        self.assertEqual(job["model"],"gemini-3.8-flash")

    def test_gemini3_suite_rejects_other_protocols(self):
        for protocol in ("anthropic","openai"):
            payload=json.dumps({"protocol":protocol,"test_kind":"gemini","base_url":"https://mux.la",
                                "api_key":"secret","model":"gemini-3.8-flash",
                                "cases":["GEMINI3_P0"]}).encode()
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
            self.assertEqual(caught.exception.code,400)

    def test_gemini_text_and_image_kinds_do_not_cross_protocols(self):
        """gemini（文本）只走 /v1beta GenerateContent 面；gemini-image 只走图像面，不可互相派发。"""
        payload=json.dumps({"protocol":"gemini","test_kind":"gemini-image",
                            "base_url":"https://mux.la","api_key":"secret",
                            "model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
                            "requests":1,"concurrency":1}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)
        # gemini 协议的 test_kinds 只放行 "gemini"（文本），gemini-image 必须被拦回，
        # 否则图像子项会被派到文本执行链。
        self.assertIn("不支持 gemini 协议", caught.exception.read().decode("utf-8"))

    def test_create_deepseek_run(self):
        payload=json.dumps({"protocol":"openai", "test_kind":"deepseek", "base_url":"https://api.deepseek.com", "api_key":"secret", "model":"deepseek-v4-pro", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "deepseek")
        job=put.call_args.args[0]
        self.assertEqual(job["case"], "DS_V4_P0")
        self.assertEqual(job["model"], "deepseek-v4-pro")

    def test_create_multi_kvv_batch(self):
        payload=json.dumps({"protocol":"volcengine","test_kind":"kvv","base_url":"https://example.com","api_key":"secret","model":"m","faces":"all","cases":["KVV_K3_FEATURES","KVV_PARAMS"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual([x["case"] for x in put.call_args.args[0]["children"]],["KVV_K3_FEATURES","KVV_PARAMS"])

    def test_create_multi_deepseek_batch(self):
        payload=json.dumps({"protocol":"openai","test_kind":"deepseek","base_url":"https://api.deepseek.com","api_key":"secret","model":"deepseek-v4-pro","cases":["DS_V4_P0","DS_V4_P1"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual([x["case"] for x in put.call_args.args[0]["children"]],["DS_V4_P0","DS_V4_P1"])

    def test_accept_versioned_deepseek_model(self):
        payload=json.dumps({"protocol":"openai", "test_kind":"deepseek", "base_url":"https://api.deepseek.com", "api_key":"secret", "model":"deepseek-v4-pro-ga-260813", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["model"],"deepseek-v4-pro-ga-260813")
        self.assertEqual(put.call_args.args[0]["model"],"deepseek-v4-pro-ga-260813")

    def test_accept_gateway_model_alias_for_deepseek_suite(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"deepseek", "base_url":"https://example.com", "api_key":"secret", "model":"ep-customer-ds-prod", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["model"],"ep-customer-ds-prod")

    def test_reject_unsupported_kvv_protocol(self):
        payload=json.dumps({"protocol":"anthropic", "test_kind":"kvv", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"KVV_PARAMS"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_create_fable_run_accepts_anthropic_protocol(self):
        # 回归：/api/run 的 test_kind→用例集派发链曾漏掉 fable，available 变成空字典，
        # 于是任何 Fable 5 提交都直接 400「未知测试类型」，套件根本发不出去。
        payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://i-token.ai","api_key":"secret","model":"claude-fable-5","cases":["FABLE5_FINGERPRINT"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"],"fable")
        job=put.call_args.args[0]
        self.assertEqual(job["case"],"FABLE5_FINGERPRINT")
        self.assertEqual(job["protocol"],"anthropic")
        self.assertEqual(job["model"],"claude-fable-5")

    def test_fable_suite_rejects_non_anthropic_protocol(self):
        payload=json.dumps({"protocol":"openai","test_kind":"fable","base_url":"https://example.com","api_key":"secret","model":"claude-fable-5","cases":["FABLE5_P0"]}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_fable_cases_are_listed_with_anthropic_protocol(self):
        cases=self.get_json("/api/cases")
        fable={x["id"]:x for x in cases if x["test_kind"]=="fable"}
        # Claude 测试：fable5 五套件按模型变体各发一份（FABLE5_* / OPUS5_* / SONNET5_* /
        # HAIKU45_* / SONNET46_*），外加四个常驻案例组（经济验证、中转计费探针、Protobuf 签名两步校验、
        # 0 注入判定；均不带 claude_variant，不随变体切换）。
        self.assertEqual(len(fable),29)
        self.assertEqual({x["claude_variant"] for x in fable.values() if x.get("claude_variant")},{"fable5","opus5","sonnet5","haiku45","sonnet46"})
        self.assertNotIn("claude_variant", fable[server.BILLING_PROBE_CASE_ID])
        self.assertTrue(fable[server.BILLING_PROBE_CASE_ID]["always_visible"])
        # 每个变体都必须提供同一组五套件（BASIC / P0 / P1 / P2 / FINGERPRINT），仅模型不同。
        for variant_id, suffix in (("fable5","FABLE5"),("opus5","OPUS5"),("sonnet5","SONNET5"),
                                   ("haiku45","HAIKU45"),("sonnet46","SONNET46")):
            self.assertEqual({x["id"] for x in fable.values() if x.get("claude_variant")==variant_id},
                             {f"{suffix}_{name}" for name in ("BASIC","P0","P1","P2","FINGERPRINT")})
        self.assertTrue(all(x["protocols"]==["anthropic"] for x in fable.values()))
        self.assertEqual(fable["FABLE5_BASIC"]["requests"],15)
        self.assertEqual(fable["FABLE5_P0"]["requests"],20)
        self.assertEqual(fable["FABLE5_P1"]["requests"],11)
        self.assertEqual(fable["FABLE5_P2"]["requests"],21)
        self.assertEqual(fable["FABLE5_FINGERPRINT"]["requests"],14)
        # 变体套件与 fable5 套件请求数一致（同构用例，仅模型不同）。
        self.assertEqual(fable["OPUS5_FINGERPRINT"]["requests"],14)
        self.assertEqual(fable["SONNET5_P2"]["requests"],21)
        self.assertEqual(fable["HAIKU45_BASIC"]["requests"],15)
        self.assertEqual(fable["HAIKU45_FINGERPRINT"]["requests"],14)
        self.assertEqual(fable["SONNET46_P0"]["requests"],20)
        self.assertEqual(fable["SONNET46_P2"]["requests"],21)
        self.assertEqual(fable["OPUS5_BASIC"]["default_model"],"claude-opus-5")
        self.assertEqual(fable["SONNET5_FINGERPRINT"]["default_model"],"claude-sonnet-5")
        self.assertEqual(fable["HAIKU45_BASIC"]["default_model"],"claude-haiku-4-5")
        self.assertEqual(fable["SONNET46_FINGERPRINT"]["default_model"],"claude-sonnet-4-6")
        self.assertIn("docs/Fable5专项测试用例设计.md", fable["FABLE5_P0"]["admission"])
        self.assertIn("Opus 5", fable["OPUS5_BASIC"]["title"])
        self.assertIn("Haiku 4.5", fable["HAIKU45_BASIC"]["title"])
        self.assertIn("Sonnet 4.6", fable["SONNET46_BASIC"]["title"])
        # 提示词、并发等展示字段与 fable5 本体一致（只换模型与标题前缀）。
        self.assertEqual(fable["HAIKU45_P1"]["concurrency"], fable["FABLE5_P1"]["concurrency"])

    def test_create_claude_variant_run_maps_to_fable5_suite(self):
        # 各模型变体用例：入队记录保留变体 ID（归档归属），派发命令映射回 FABLE5_* 套件。
        payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://i-token.ai","api_key":"secret","model":"claude-opus-5","cases":["OPUS5_FINGERPRINT"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"],"fable")
        job=put.call_args.args[0]
        self.assertEqual(job["case"],"OPUS5_FINGERPRINT")
        self.assertEqual(job["model"],"claude-opus-5")
        self.assertEqual(server.claude_suite_name(job["case"]),"FABLE5_FINGERPRINT")

    def test_create_haiku_and_sonnet46_variant_runs_map_to_fable5_suites(self):
        # Haiku 4.5 / Sonnet 4.6 与既有变体同构：变体 ID → FABLE5_* 套件名，模型按提交值下发。
        for case_id, model, suite in (("HAIKU45_P2","claude-haiku-4-5","FABLE5_P2"),
                                      ("HAIKU45_BASIC","claude-haiku-4-5","FABLE5_BASIC"),
                                      ("SONNET46_FINGERPRINT","claude-sonnet-4-6","FABLE5_FINGERPRINT")):
            payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://i-token.ai",
                                "api_key":"secret","model":model,"cases":[case_id]}).encode()
            with patch.object(server.JOB_QUEUE,"put") as put:
                with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                    result=json.load(response)
            self.assertEqual(result["test_kind"],"fable")
            job=put.call_args.args[0]
            self.assertEqual(job["case"],case_id)
            self.assertEqual(job["model"],model)
            self.assertEqual(job["protocol"],"anthropic")
            self.assertEqual(server.claude_suite_name(job["case"]),suite)

    def test_create_multi_load_batch(self):
        payload=json.dumps({"protocol":"volcengine","test_kind":"load","base_url":"https://example.com","api_key":"secret","model":"m","cases":["T0","B1"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual(batch["case_ids"],["T0","B1"])
        job=put.call_args.args[0]
        self.assertTrue(job["is_batch"])
        self.assertEqual([x["case"] for x in job["children"]],["T0","B1"])
        detail=self.get_json(f"/api/runs/{batch['id']}")
        self.assertEqual(len(detail["children"]),2)

    def test_create_proto_signature_multi_model_batch(self):
        """执行范围多选：同用例 × 多模型拆成批，每个模型一条子记录，父记录 model 存摘要。"""
        payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://example.com",
                            "api_key":"secret","model":"claude-fable-5","case":"PROTOBUF_SIGNATURE_CHECK",
                            "case_models":["claude-fable-5","claude-opus-5","claude-fable-5","claude-sonnet-5"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual(batch["model"],"3 个模型")
        job=put.call_args.args[0]
        self.assertTrue(job["is_batch"])
        # 去重且保序：重复提交的模型只跑一次。
        self.assertEqual([x["model"] for x in job["children"]],["claude-fable-5","claude-opus-5","claude-sonnet-5"])
        self.assertTrue(all(x["case"]=="PROTOBUF_SIGNATURE_CHECK" for x in job["children"]))
        detail=self.get_json(f"/api/runs/{batch['id']}")
        self.assertEqual([child["model"] for child in detail["children"]],["claude-fable-5","claude-opus-5","claude-sonnet-5"])

    def test_proto_signature_scope_single_or_missing_stays_single_run(self):
        """执行范围只认 ≥2 个模型：单选/未选走原有单模型链路，行为不变。"""
        base={"protocol":"anthropic","test_kind":"fable","base_url":"https://example.com",
              "api_key":"secret","model":"claude-fable-5","case":"PROTOBUF_SIGNATURE_CHECK"}
        for scope in (["claude-opus-5"], None):
            payload=json.dumps({**base, **({"case_models":scope} if scope else {})}).encode()
            with patch.object(server.JOB_QUEUE,"put") as put:
                with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                    result=json.load(response)
            self.assertFalse(result["is_batch"])
            self.assertEqual(result["model"],"claude-fable-5")
            self.assertEqual(put.call_args.args[0]["model"],"claude-fable-5")

    def test_case_models_ignored_for_other_fable_cases(self):
        """执行范围只对签名校验生效：其它用例即使带上 case_models 也按单模型跑。"""
        payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://example.com",
                            "api_key":"secret","model":"claude-fable-5","case":"ZERO_INJECTION_CHECK",
                            "case_models":["claude-opus-5","claude-sonnet-5"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertFalse(result["is_batch"])
        self.assertEqual(put.call_args.args[0]["model"],"claude-fable-5")

    def test_cases_payload_exposes_model_scope_for_signature(self):
        cases=self.get_json("/api/cases")
        signature=next(item for item in cases if item["id"]=="PROTOBUF_SIGNATURE_CHECK")
        self.assertTrue(signature.get("model_scope"))
        self.assertTrue(signature.get("model_scope_hint"))
        # GLM 家族复用同一套卡片，允许三款型号分别执行并出报告。
        glm = next(item for item in cases if item["id"]=="GLM_53_FULL")
        self.assertTrue(glm.get("model_scope"))
        self.assertEqual([p["id"] for p in glm["model_profiles"]], ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"])

    def test_comprehensive_cards_split_requested_models_into_independent_jobs(self):
        for kind, card, models in (
            ("glm","GLM_53_FULL",["glm-5.3","glm-5.3-flash","glm-5.3-flashx"]),
            ("qwen","QWEN_FULL",["qwen3.8-max","qwen3.7-max"]),
        ):
            payload=json.dumps({"protocol":"openai","test_kind":kind,"base_url":"https://example.com",
                                "api_key":"secret","model":models[0],"case":card,"case_models":models}).encode()
            with patch.object(server.JOB_QUEUE,"put") as put:
                with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                    result=json.load(response)
            self.assertTrue(result["is_batch"])
            self.assertEqual([child["model"] for child in put.call_args.args[0]["children"]],models)

    def test_connectivity_check(self):
        payload=json.dumps({"protocol":"volcengine", "base_url":"https://example.com", "api_key":"secret", "model":"m"}).encode()
        result={"name":"connectivity","passed":True,"http_status":200,"duration_ms":12.3}
        with patch.object(server, "check_connectivity", return_value=result) as checked:
            with urlopen(Request(self.base+"/api/connectivity/check",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                actual=json.load(response)
        self.assertTrue(actual["passed"])
        config=checked.call_args.args[0]
        self.assertEqual(config.protocol,"volcengine")
        self.assertEqual(config.api_key,"secret")

    def test_connectivity_check_shares_trust_env_switch_with_load_tests(self):
        """探针必须与正式压测共用 VOLC_TRUST_ENV。

        否则会出现"模型列表查得到、连通性检查却走直连"的前后不一致：
        探针忽略 .env 里的代理设置，直连在境外网关上失败，正式测试根本起不来。
        """
        payload=json.dumps({"protocol":"anthropic", "base_url":"https://mux.la", "api_key":"secret", "model":"kimi-k3"}).encode()
        result={"name":"connectivity","passed":True,"http_status":200,"duration_ms":12.3}
        for flag, expected in ((True, True), (False, False)):
            with patch.object(server, "trust_env_enabled", return_value=flag):
                with patch.object(server, "check_connectivity", return_value=result) as checked:
                    with urlopen(Request(self.base+"/api/connectivity/check",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                        json.load(response)
            self.assertIs(checked.call_args.args[0].trust_env, expected)

    def test_query_models(self):
        payload=json.dumps({"protocol":"openai", "base_url":"https://example.com", "api_key":"secret"}).encode()
        fake = type("Response", (), {"status_code": 200, "text": "", "json": lambda self: {"data": [{"id":"model-b"},{"id":"model-a"}]}})()
        with patch.object(server.httpx, "get", return_value=fake) as get:
            with urlopen(Request(self.base+"/api/models/query",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual([m["id"] for m in result["models"]], ["model-a", "model-b"])
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(get.call_args.args[0], "https://example.com/v1/models")
        # 与探针保持一致：是否走环境代理由 VOLC_TRUST_ENV 决定，不再隐式使用。
        self.assertIn("trust_env", get.call_args.kwargs)

    def test_reject_bad_url(self):
        payload=json.dumps({"base_url":"bad", "api_key":"x", "model":"m", "case":"T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_reject_base_url_with_path(self):
        payload=json.dumps({"protocol":"openai", "base_url":"https://mux.la/v1", "api_key":"x", "model":"m", "case":"T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def _make_image_run(self, results: Path, run_id: str) -> Path:
        """造一条带 images/ 的图片测试记录，返回结果目录。"""
        result_dir = results / f"GEMINI_IMAGE_GEN-20261008-170000-{run_id[:6]}"
        (result_dir / "images").mkdir(parents=True)
        (result_dir / "summary.json").write_text('{"admission_passed": true}', encoding="utf-8")
        (result_dir / "requests.jsonl").write_text('{"index": 1, "images": []}\n', encoding="utf-8")
        (result_dir / "images" / "req-0001-01.png").write_bytes(b"\x89PNG\r\n\x1a\n-fake-image")
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, "gemini-image", "GEMINI_IMAGE_GEN", "gemini_images", "https://mux.la",
                 "gemini-3-pro-image", "completed", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), result_dir.name),
            )
        return result_dir

    def test_generated_image_gallery_shows_all_images_and_serves_originals(self):
        run_id = f"gallery-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            target = self._make_image_run(results, run_id)
            for index in range(2, 28):
                (target / "images" / f"req-{index:04d}-01.png").write_bytes(b"image")
            special_name = 'image-<&"#.webp'
            (target / "images" / special_name).write_bytes(b"webp-image")
            (target / "images" / "notes.txt").write_text("not an image")
            (target / "images" / "nested.png").mkdir()
            try:
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    for test_kind in ("gemini-image", "gpt-image"):
                        with server.db() as connection:
                            connection.execute("UPDATE runs SET test_kind=?,model=? WHERE id=?",
                                               (test_kind, '<img src=x onerror="alert(1)">', run_id))
                        detail = self.get_json(f"/api/runs/{run_id}")
                        self.assertEqual(detail["images_url"], f"/api/runs/{run_id}/images")
                        listing = self.get_json(f"/api/runs?q={run_id}")["items"]
                        self.assertEqual(listing[0]["images_url"], detail["images_url"])
                        with urlopen(self.base + detail["images_url"]) as response:
                            self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")
                            page = response.read().decode()
                        self.assertEqual(page.count('<figure '), 28)
                        self.assertIn("共 28 张图片", page)
                        self.assertLess(page.index("req-0002-01.png"), page.index("req-0027-01.png"))
                        self.assertNotIn("notes.txt", page)
                        self.assertNotIn("nested.png", page)
                        self.assertNotIn('<img src=x onerror="alert(1)">', page)
                        self.assertIn("&lt;img src=x", page)
                        self.assertIn(quote(special_name, safe=""), page)
                        with urlopen(self.base + detail["images_url"] + "/req-0001-01.png") as response:
                            self.assertEqual(response.headers["Content-Type"], "image/png")
                            self.assertEqual(response.read(), (target / "images" / "req-0001-01.png").read_bytes())
                        with urlopen(self.base + detail["images_url"] + "/" + quote(special_name, safe="")) as response:
                            self.assertEqual(response.headers["Content-Type"], "image/webp")
                            self.assertEqual(response.read(), b"webp-image")
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_generated_image_gallery_empty_and_path_guards(self):
        run_id = f"gallery-guard-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            target = self._make_image_run(results, run_id)
            image_dir = target / "images"
            outside = results / "outside.png"
            outside.write_bytes(b"private-image")
            (image_dir / "linked.png").symlink_to(outside)
            (image_dir / "summary.png").symlink_to(target / "summary.json")
            (image_dir / "notes.txt").write_text("private text")
            try:
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    for name in ("missing.png", "notes.txt", "linked.png", "summary.png",
                                 "..%2Fsummary.json", "%2E%2E%2F%2E%2E%2Foutside.png"):
                        with self.assertRaises(HTTPError) as caught:
                            urlopen(f"{self.base}/api/runs/{run_id}/images/{name}")
                        self.assertEqual(caught.exception.code, 404, name)
                    (image_dir / "req-0001-01.png").unlink()
                    self.assertIsNone(self.get_json(f"/api/runs/{run_id}")["images_url"])
                    with urlopen(f"{self.base}/api/runs/{run_id}/images") as response:
                        page = response.read().decode()
                    self.assertIn("暂无已保存的图片", page)
                    self.assertNotIn('<figure ', page)
                    with server.db() as connection:
                        connection.execute("UPDATE runs SET result_dir=? WHERE id=?", (str(results.parent), run_id))
                    self.assertIsNone(self.get_json(f"/api/runs/{run_id}")["images_url"])
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(f"{self.base}/api/runs/{run_id}/images/outside.png")
                    self.assertEqual(caught.exception.code, 404)
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_download_bundle_keeps_generated_images(self):
        run_id = f"download-images-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            target = self._make_image_run(results, run_id)
            try:
                # ROOT 与 RESULTS 一起指向临时目录，result_dir 用目录名（相对 ROOT）。
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    with urlopen(f"{self.base}/api/runs/{run_id}/download") as response:
                        self.assertEqual(response.headers["Content-Type"], "application/zip")
                        payload = response.read()
                names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
                # 服务端在响应体写完后才删除临时归档，客户端可能先读到 EOF。
                for _ in range(50):
                    leftovers = list(results.glob(".download-*.zip"))
                    if not leftovers:
                        break
                    time.sleep(0.02)
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.assertIn("package-metadata.json", names)
        self.assertIn(f"{target.name}/images/req-0001-01.png", names)
        self.assertIn(f"{target.name}/requests.jsonl", names)
        # 临时归档必须清理，不能留在结果目录里。
        self.assertEqual(leftovers, [])

    def test_selected_download_bundle_keeps_generated_images(self):
        run_id = f"selected-images-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            target = self._make_image_run(results, run_id)
            try:
                payload = json.dumps({"run_ids": [run_id]}).encode()
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    with urlopen(Request(self.base + "/api/runs/download", data=payload, method="POST",
                                         headers={"Content-Type": "application/json"})) as response:
                        data = response.read()
                bundle = zipfile.ZipFile(io.BytesIO(data))
                names = bundle.namelist()
                metadata = json.loads(bundle.read("package-metadata.json"))
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.assertEqual(metadata["type"], "selected-runs")
        self.assertEqual(metadata["selected_run_ids"], [run_id])
        self.assertIn(f"{target.name}/images/req-0001-01.png", names)

    def _kvv_temp_payload(self, *, protocol="volcengine", temporary_case="CACHE-HIT-7", model="kimi-k3"):
        return json.dumps({
            "protocol": protocol, "test_kind": "kvv", "base_url": "https://i-token.ai",
            "api_key": "secret", "model": model, "case": "K3_TEMP_SUITE", "temporary_case": temporary_case,
        }).encode()

    def _billing_payload(self, *, protocol="anthropic", billing_probe="BILLING-ALL", model="claude-fable-5"):
        return json.dumps({
            "protocol": protocol, "test_kind": "fable", "base_url": "https://i-token.ai",
            "api_key": "secret", "model": model, "case": server.BILLING_PROBE_CASE_ID,
            "billing_probe": billing_probe,
        }).encode()

    def test_create_billing_probe_run_under_claude_module(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=self._billing_payload(),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["test_kind"], "fable")
        self.assertEqual(result["case_name"], "BILLING-ALL")
        self.assertEqual(result["case_title"], "中转计费探针 · 全部 11 条")
        job = put.call_args.args[0]
        self.assertEqual(job["case"], server.BILLING_PROBE_CASE_ID)
        self.assertEqual(job["billing_probe"], "BILLING-ALL")
        self.assertEqual(job["protocol"], "anthropic")

    def test_create_billing_probe_single_case(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs",
                                 data=self._billing_payload(billing_probe="BILLING-B02"),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "BILLING-B02")
        self.assertEqual(result["case_title"], "预扣结算口径与释放时延（是否退回）")
        self.assertEqual(put.call_args.args[0]["billing_probe"], "BILLING-B02")

    def test_billing_probe_defaults_to_whole_group(self):
        # 选中计费卡片但未指定子项时按整组处理，避免「选了卡片却跑不动」。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": [server.BILLING_PROBE_CASE_ID]}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "BILLING-ALL")
        self.assertEqual(put.call_args.args[0]["billing_probe"], "BILLING-ALL")

    def test_billing_probe_rejects_unknown_selection(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._billing_payload(billing_probe="BILLING-NOPE"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("中转计费探针", json.loads(caught.exception.read())["error"])

    def test_billing_probe_rejects_other_protocol(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._billing_payload(protocol="gemini", model="gemini-2.5-flash"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("Anthropic", json.loads(caught.exception.read())["error"])

    def test_billing_probe_ignored_for_plain_claude_suite(self):
        # 普通 Claude 套件即便带了 billing_probe 残留，也必须置空并走 fable5 执行器。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5", "cases": ["FABLE5_P0"],
                              "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "FABLE5_P0")
        self.assertIsNone(put.call_args.args[0]["billing_probe"])

    def test_billing_probe_in_batch_only_probe_child_carries_selection(self):
        # 计费探针在 Claude tab 下常与各套件同批提交。旧逻辑要求 case_names 恰好只有探针
        # 一条才保留 billing_probe，批量时被整批置空 → 探针子任务落到 run_fable5.py 报
        # invalid choice（历史批次因此整批不准入）。批量下必须逐子任务区分。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-haiku-4-5",
                              "cases": ["HAIKU45_BASIC", server.BILLING_PROBE_CASE_ID],
                              "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertTrue(result["is_batch"])
        children = put.call_args.args[0]["children"]
        self.assertEqual([child["case"] for child in children], ["HAIKU45_BASIC", server.BILLING_PROBE_CASE_ID])
        self.assertIsNone(children[0]["billing_probe"])
        self.assertEqual(children[1]["billing_probe"], "BILLING-ALL")
        # 探针子记录的显示名按所选子项归档，与单条运行口径一致（记录页显示"全部 11 条"）。
        detail = self.get_json(f"/api/runs/{result['id']}")
        titles = {row["case_name"]: row.get("case_title") for row in detail["children"]}
        self.assertEqual(titles["BILLING-ALL"], "中转计费探针 · 全部 11 条")
        self.assertIsNone(titles["HAIKU45_BASIC"])

    def test_billing_probe_in_batch_passes_single_probe_id(self):
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": ["FABLE5_P1", server.BILLING_PROBE_CASE_ID],
                              "billing_probe": "BILLING-U03"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        children = {child["case"]: child for child in put.call_args.args[0]["children"]}
        self.assertEqual(children[server.BILLING_PROBE_CASE_ID]["billing_probe"], "BILLING-U03")
        self.assertIsNone(children["FABLE5_P1"]["billing_probe"])

    def test_batch_with_billing_probe_rejects_non_anthropic_protocol(self):
        # 探针只走 Anthropic Messages 面；同批混入时也必须拦回，而不是把探针派给错的面。
        payload = json.dumps({"protocol": "openai", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": ["FABLE5_P1", server.BILLING_PROBE_CASE_ID]}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs", data=payload,
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("Anthropic", json.loads(caught.exception.read())["error"])

    def test_batch_without_billing_probe_keeps_selection_empty(self):
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": ["FABLE5_BASIC", "FABLE5_P0"], "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        children = put.call_args.args[0]["children"]
        self.assertTrue(all(child["billing_probe"] is None for child in children))

    def test_kvv_temp_anthropic_is_rejected(self):
        # anthropic 面下 K3 案例不可执行（K3 走 Chat Completions）；计费探针在 Claude 模块。
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._kvv_temp_payload(protocol="anthropic", model="claude-fable-5"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("openai", json.loads(caught.exception.read())["error"])

    def test_kvv_temp_rejects_billing_selection(self):
        # 计费探针搬走后，K3 案例集不再接受 BILLING-* 选择。
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._kvv_temp_payload(temporary_case="BILLING-ALL"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("未知 K3 临时测试案例", json.loads(caught.exception.read())["error"])

    def test_kvv_temp_k3_selection_still_accepted_on_chat_completions(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=self._kvv_temp_payload(),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "CACHE-HIT-7")
        self.assertEqual(result["test_kind"], "kvv")
        self.assertEqual(put.call_args.args[0]["temporary_case"], "CACHE-HIT-7")

    def test_kvv_temp_cannot_be_batched_with_official_suites(self):
        # K3 案例集是 24/68 条的大套件，不与官方 KVV 套件同批提交。
        payload = json.dumps({"protocol": "volcengine", "test_kind": "kvv", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "kimi-k3", "faces": "all",
                              "cases": ["K3_TEMP_SUITE", "KVV_PARAMS"]}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs", data=payload,
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("只能单独运行", json.loads(caught.exception.read())["error"])

    def test_billing_probe_is_listed_under_claude_module(self):
        cases = self.get_json("/api/cases")
        billing = [item for item in cases if item["id"] == server.BILLING_PROBE_CASE_ID]
        self.assertEqual(len(billing), 1)
        item = billing[0]
        self.assertEqual(item["test_kind"], "fable")
        self.assertTrue(item["always_visible"])
        self.assertTrue(item["billing_probe"])
        self.assertEqual(item["protocols"], ["anthropic"])
        self.assertEqual({option.get("group") for option in item["case_options"]}, {"中转计费探针"})
        self.assertEqual(item["case_count"], 12)
        # K3 临时案例集现在挂在「Kimi 测试」（kvv）模块下，不再暴露计费分组。
        kvv_temp = [entry for entry in cases if entry["id"] == server.KVV_TEMP_CASE_ID][0]
        self.assertEqual(kvv_temp["test_kind"], "kvv")
        self.assertEqual({option.get("group") for option in kvv_temp["case_options"]}, {"K3 供应商自测基线"})
        self.assertEqual(kvv_temp["case_count"], 24)

    def test_model_name_check_is_removed_from_claude_cases(self):
        """已下线的模型名称校验不再出现在 Claude 用例列表，也不能新建任务。"""
        cases = self.get_json("/api/cases")
        self.assertFalse(any(item["id"] == "MODEL_NAME_CHECK" for item in cases))
        self.assertNotIn("MODEL_NAME_CHECK", server.CLAUDE_RUNNABLE_CASES)
        payload = json.dumps({
            "protocol": "anthropic", "test_kind": "fable",
            "base_url": "https://example.com", "api_key": "secret",
            "model": "claude-opus-4-7", "case": "MODEL_NAME_CHECK",
        }).encode()
        status, result = self._post_run(payload)
        self.assertEqual(status, 400)
        self.assertIn("未知测试类型", result["error"])

    def test_proto_signature_card_and_selected_models(self):
        items = [item for item in self.get_json("/api/cases") if item["id"] == server.PROTO_SIGNATURE_CASE_ID]
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["always_visible"])
        self.assertTrue(items[0]["show_expected_model"])
        self.assertNotIn("claude_variant", items[0])
        self.assertEqual(items[0]["requests"], 4)
        self.assertEqual(items[0]["case_count"], 3)
        self.assertEqual(items[0]["protocols"], ["anthropic"])
        # 输出预算默认给足（给少了思考吃满上限 ⇒ 正文为零 ⇒ 假 failed），界面可覆盖。
        self.assertEqual(items[0]["output_tokens"], server.SIGNATURE_DEFAULT_MAX_TOKENS)
        self.assertEqual(items[0]["signature_max_tokens_default"], server.SIGNATURE_DEFAULT_MAX_TOKENS)
        self.assertEqual(items[0]["signature_max_tokens_limit"], server.SIGNATURE_MAX_MAX_TOKENS)
        # thinking 档位按模型族解析：4.5/4.6 系 enabled、5 系 adaptive_effort、fable 系 adaptive。
        expected_modes = {"fable5": "adaptive", "opus5": "adaptive_effort",
                          "sonnet5": "adaptive_effort", "haiku45": "enabled",
                          "sonnet46": "enabled"}
        for variant, (_label, model) in server.CLAUDE_VARIANTS.items():
            with self.subTest(variant=variant):
                captured = self._execute_child_capture({
                    "test_kind": "fable", "case": server.PROTO_SIGNATURE_CASE_ID,
                    "protocol": "anthropic", "model": model, "claude_variant": variant,
                })
                command = [str(part) for part in captured["command"]]
                self.assertEqual(Path(command[1]).name, "proto_signature_check.py")
                self.assertEqual(command[command.index("--model") + 1], model)
                self.assertEqual(command[command.index("--thinking-mode") + 1],
                                 expected_modes[variant])
                self.assertEqual(command[command.index("--max-tokens") + 1],
                                 str(server.SIGNATURE_DEFAULT_MAX_TOKENS))
                self.assertIn("--output-root", command)

    def test_proto_signature_output_budget_is_passed_and_validated(self):
        # 界面可覆盖预算，值必须落到命令行。
        captured = self._execute_child_capture({
            "test_kind": "fable", "case": server.PROTO_SIGNATURE_CASE_ID,
            "protocol": "anthropic", "model": "claude-sonnet-4-6", "signature_max_tokens": 8192,
        })
        command = [str(part) for part in captured["command"]]
        self.assertEqual(command[command.index("--max-tokens") + 1], "8192")

        # 越界/非整数在提交期就拦住，不烧请求。
        for bad in (0, -1, server.SIGNATURE_MAX_MAX_TOKENS + 1, "big", True):
            with self.subTest(bad=bad):
                payload = json.dumps({"protocol": "anthropic", "test_kind": "fable",
                                      "base_url": "https://example.com", "api_key": "secret",
                                      "model": "claude-sonnet-4-6",
                                      "cases": [server.PROTO_SIGNATURE_CASE_ID],
                                      "signature_max_tokens": bad}).encode()
                with patch.object(server, "start_worker"), patch.object(server.JOB_QUEUE, "put"):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                        headers={"Content-Type": "application/json"}))
                self.assertEqual(caught.exception.code, 400)

    def test_signature_budget_is_dropped_for_other_cases(self):
        # 界面残留的预算不得被静默带进其他套件。
        for cases in (["HAIKU45_BASIC"], ["OPUS5_BASIC"]):
            with self.subTest(cases=cases):
                payload = json.dumps({"protocol": "anthropic", "test_kind": "fable",
                                      "base_url": "https://example.com", "api_key": "secret",
                                      "model": "claude-haiku-4-5", "cases": cases,
                                      "signature_max_tokens": 32000}).encode()
                with patch.object(server, "start_worker"), patch.object(server.JOB_QUEUE, "put") as put:
                    with urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                         headers={"Content-Type": "application/json"})) as response:
                        self.assertEqual(response.status, 202)
                queued = put.call_args.args[0]
                for job in queued.get("children", [queued]):
                    self.assertIsNone(job.get("signature_max_tokens"))

    def test_proto_signature_date_version_of_haiku_uses_enabled_thinking(self):
        captured = self._execute_child_capture({
            "test_kind": "fable", "case": server.PROTO_SIGNATURE_CASE_ID,
            "protocol": "anthropic", "model": "claude-haiku-4-5-20251001",
        })
        command = [str(part) for part in captured["command"]]
        self.assertEqual(command[command.index("--thinking-mode") + 1], "enabled")

    def test_zero_injection_card_and_dispatch(self):
        """0 注入判定是 Claude 模块下的独立常驻案例组（四条用例，自包含 CLI）。"""
        items = [item for item in self.get_json("/api/cases") if item["id"] == server.ZERO_INJECTION_CASE_ID]
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["test_kind"], "fable")
        self.assertTrue(item["always_visible"])
        self.assertTrue(item["show_expected_model"])
        self.assertNotIn("claude_variant", item)
        self.assertEqual(item["protocols"], ["anthropic"])
        self.assertEqual(item["executor"], "fable5/zero_injection_check.py")
        self.assertEqual(item["case_count"], 4)
        self.assertNotIn("case_options", item)
        self.assertIn(server.ZERO_INJECTION_CASE_ID, server.CLAUDE_RUNNABLE_CASES)
        # 非 anthropic 协议必须被拦回（本组只在 Messages 面取证）
        payload = json.dumps({"protocol": "openai", "test_kind": "fable", "base_url": "https://example.com",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": [server.ZERO_INJECTION_CASE_ID]}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                            headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        # 五个 Claude 变体共用同一执行器，thinking 档位按被测模型族解析（与签名校验同一权威）
        expected_modes = {"fable5": "adaptive", "opus5": "adaptive_effort",
                          "sonnet5": "adaptive_effort", "haiku45": "enabled",
                          "sonnet46": "enabled"}
        for variant, (_label, model) in server.CLAUDE_VARIANTS.items():
            with self.subTest(variant=variant):
                captured = self._execute_child_capture({
                    "test_kind": "fable", "case": server.ZERO_INJECTION_CASE_ID,
                    "protocol": "anthropic", "model": model, "claude_variant": variant,
                })
                command = [str(part) for part in captured["command"]]
                self.assertEqual(Path(command[1]).name, "zero_injection_check.py")
                self.assertEqual(command[command.index("--model") + 1], model)
                self.assertEqual(command[command.index("--thinking-mode") + 1],
                                 expected_modes[variant])
                self.assertIn("--output-root", command)

    def test_proto_signature_run_preserves_variant_for_gateway_alias(self):
        for cases in ([server.PROTO_SIGNATURE_CASE_ID], ["HAIKU45_BASIC", server.PROTO_SIGNATURE_CASE_ID]):
            payload = json.dumps({"protocol": "anthropic", "test_kind": "fable",
                                  "base_url": "https://example.com", "api_key": "secret",
                                  "model": "relay-haiku", "claude_variant": "haiku45", "cases": cases}).encode()
            with patch.object(server, "start_worker"), patch.object(server.JOB_QUEUE, "put") as put:
                with urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                     headers={"Content-Type": "application/json"})) as response:
                    self.assertEqual(response.status, 202)
            queued = put.call_args.args[0]
            jobs = queued.get("children", [queued])
            for job in jobs:
                self.assertEqual(job["model"], "relay-haiku")
                self.assertEqual(job["claude_variant"], "haiku45")
        captured = self._execute_child_capture({"test_kind": "fable", "case": server.PROTO_SIGNATURE_CASE_ID,
                                                "protocol": "anthropic", "model": "relay-haiku",
                                                "claude_variant": "haiku45"})
        command = [str(part) for part in captured["command"]]
        self.assertEqual(command[command.index("--model") + 1], "relay-haiku")
        self.assertEqual(command[command.index("--thinking-mode") + 1], "enabled")

    def _execute_child_capture(self, job_fields):
        run_id = f"child-{time.time_ns()}"
        stored = job_fields.get("billing_probe") or job_fields.get("temporary_case") or job_fields["case"]
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, job_fields["test_kind"], stored, job_fields["protocol"], "https://i-token.ai",
                 "claude-fable-5", "queued", 0, time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
            )
        job = {"id": run_id, "base_url": "https://i-token.ai", "api_key": "sk-x",
               "model": "claude-fable-5", "quick": False, **job_fields}
        captured = {}

        class FakeProcess:
            def wait(self):
                return 0

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs.get("env")
            return FakeProcess()

        try:
            with tempfile.TemporaryDirectory(dir=server.ROOT) as directory:
                with patch.object(server, "RESULTS", Path(directory)), \
                     patch.object(server, "latest_result", return_value=None), \
                     patch.object(server.subprocess, "Popen", fake_popen):
                    server._execute_child(job)
        finally:
            with server.db() as connection:
                connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return captured

    def _run_billing_child(self, billing_probe, protocol="anthropic"):
        return self._execute_child_capture({
            "test_kind": "fable", "case": server.BILLING_PROBE_CASE_ID,
            "protocol": protocol, "billing_probe": billing_probe,
        })

    def test_batch_children_dispatch_to_matching_runners(self):
        # 端到端：批量里同一批子任务按各自 case 派到不同执行器（探针→计费脚本，套件→fable5）。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-haiku-4-5",
                              "cases": ["HAIKU45_BASIC", server.BILLING_PROBE_CASE_ID],
                              "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        children = {child["case"]: child for child in put.call_args.args[0]["children"]}
        probe = self._execute_child_capture({
            "test_kind": "fable", "case": server.BILLING_PROBE_CASE_ID, "protocol": "anthropic",
            "billing_probe": children[server.BILLING_PROBE_CASE_ID]["billing_probe"]})
        self.assertEqual(Path(probe["command"][1]).name, "run_relay_billing_probe.py")
        self.assertEqual(probe["env"]["BILLING_PROBES"], "")
        suite = self._execute_child_capture({
            "test_kind": "fable", "case": "HAIKU45_BASIC", "protocol": "anthropic", "billing_probe": None})
        self.assertEqual(Path(suite["command"][1]).name, "run_fable5.py")
        # 套件子任务用 fable5 真实套件名做位置参数，且不带任何计费环境变量。
        self.assertEqual(suite["command"][2], "FABLE5_BASIC")
        self.assertNotIn("BILLING_PROBES", suite["env"])

    def _run_temp_child(self, temporary_case, protocol="volcengine"):
        return self._execute_child_capture({
            "test_kind": "kvv", "case": temporary_case,
            "protocol": protocol, "temporary_case": temporary_case,
        })

    def test_billing_child_uses_probe_runner_and_billing_env(self):
        captured = self._run_billing_child("BILLING-ALL")
        self.assertEqual(Path(captured["command"][1]).name, "run_relay_billing_probe.py")
        # 计费探针靠环境变量驱动，命令里不得出现位置参数。
        self.assertNotIn("BILLING-ALL", [str(part) for part in captured["command"]])
        env = captured["env"]
        self.assertEqual(env["BILLING_BASE_URL"], "https://i-token.ai")
        self.assertEqual(env["BILLING_API_KEY"], "sk-x")
        self.assertEqual(env["BILLING_MODEL"], "claude-fable-5")
        self.assertEqual(env["BILLING_PROTOCOL"], "anthropic")
        # 整组 = 空（执行器按「空即全部 11 条」处理）。
        self.assertEqual(env["BILLING_PROBES"], "")

    def test_billing_child_passes_selected_probe_ids(self):
        captured = self._run_billing_child("BILLING-U03")
        self.assertEqual(captured["env"]["BILLING_PROBES"], "BILLING-U03")
        self.assertEqual(captured["env"]["BILLING_PROTOCOL"], "anthropic")

    def test_billing_child_maps_openai_protocol_to_chat_completions(self):
        # 探针脚本只认 anthropic / openai 两种面；其它协议在入口层已被拒绝。
        captured = self._run_billing_child("BILLING-S01", protocol="openai")
        self.assertEqual(captured["env"]["BILLING_PROTOCOL"], "openai")
        self.assertEqual(captured["env"]["BILLING_BASE_URL"], "https://i-token.ai")

    def test_path_override_flows_into_runner_env(self):
        # 自定义请求路径：入库存原始值，执行时剥掉操作后缀拼进 VOLC/FABLE 系 base URL。
        payload = json.dumps({"protocol": "volcengine", "test_kind": "load",
                              "base_url": "https://relay.example", "api_key": "secret",
                              "model": "ep-x", "cases": ["SIMPLE_LOAD"],
                              "path_override": "/api/v3/contents/generations/tasks"}).encode()
        with patch.object(server, "start_worker"), patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                self.assertEqual(response.status, 202)
        job = put.call_args.args[0]
        # 负载套件可能走批量分支：字段挂在每个子任务上（单任务分支挂在 job 本身）。
        children = job.get("children") or [job]
        self.assertTrue(all(child["path_override"] == "/api/v3/contents/generations/tasks" for child in children))
        with server.db() as connection:
            row = connection.execute("SELECT path_override FROM runs WHERE id=?", (job["id"],)).fetchone()
        self.assertEqual(row["path_override"], "/api/v3/contents/generations/tasks")
        captured = self._execute_child_capture({"test_kind": "fable", "case": "HAIKU45_BASIC",
                                                "protocol": "anthropic", "billing_probe": None,
                                                "path_override": "/relay/anthropic"})
        self.assertEqual(captured["env"]["FABLE_BASE_URL"], "https://i-token.ai/relay/anthropic")

    def test_path_override_keeps_billing_probe_base_url_host_only(self):
        # 计费探针自行拼请求路径：BILLING_BASE_URL 必须保持 host-only，不掺自定义前缀。
        captured = self._execute_child_capture({"test_kind": "fable", "case": server.BILLING_PROBE_CASE_ID,
                                                "protocol": "anthropic", "billing_probe": "BILLING-ALL",
                                                "path_override": "/relay/anthropic"})
        self.assertEqual(captured["env"]["BILLING_BASE_URL"], "https://i-token.ai")

    def test_path_override_validated_at_entry(self):
        payload = json.dumps({"protocol": "openai", "test_kind": "load",
                              "base_url": "https://relay.example", "api_key": "secret",
                              "model": "gpt-x", "cases": ["SIMPLE_LOAD"],
                              "path_override": "api/v3"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                            headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("以 / 开头", json.load(caught.exception)["error"])

    def test_k3_child_keeps_manifest_runner_without_billing_env(self):
        captured = self._run_temp_child("CACHE-HIT-7")
        self.assertEqual(Path(captured["command"][1]).name, "run_kvv_temp.py")
        self.assertEqual(captured["env"]["VOLC_TEMP_CASE"], "CACHE-HIT-7")
        self.assertNotIn("BILLING_PROBES", captured["env"])

    def test_k3_child_running_whole_suite_uses_entry_case_id(self):
        # 整组运行时 job["case"] 就是入口 id，子项选择为空（执行器按「空 = 全部合并案例」处理）。
        captured = self._execute_child_capture({
            "test_kind": "kvv", "case": server.KVV_TEMP_CASE_ID,
            "protocol": "volcengine", "temporary_case": None, "billing_probe": None,
        })
        self.assertEqual(Path(captured["command"][1]).name, "run_kvv_temp.py")
        self.assertEqual(captured["env"]["VOLC_TEMP_CASE"], "")
        self.assertNotIn(server.KVV_TEMP_CASE_ID, [str(part) for part in captured["command"]])

    def _chat_payload(self, **overrides):
        body = {
            "protocol": "openai", "test_kind": "chat", "base_url": "https://mux.la",
            "api_key": "secret", "model": "gpt-5.6", "case": "CHAT_PROMPT",
            "chat_prompt": "你给我画一个鹈鹕骑自行车的 SVG",
            "chat_system": "你是 SVG 生成器", "chat_max_tokens": 4096, "chat_temperature": 0.4,
        }
        body.update(overrides)
        return json.dumps(body).encode()

    def test_create_chat_run_keeps_prompt_and_options(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=self._chat_payload(), method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["test_kind"], "chat")
        self.assertEqual(result["case_name"], "CHAT_PROMPT")
        self.assertFalse(result["is_batch"])
        job = put.call_args.args[0]
        self.assertEqual(job["case"], "CHAT_PROMPT")
        self.assertEqual(job["chat_prompt"], "你给我画一个鹈鹕骑自行车的 SVG")
        self.assertEqual(job["chat_system"], "你是 SVG 生成器")
        self.assertEqual(job["chat_max_tokens"], 4096)
        self.assertEqual(job["chat_temperature"], 0.4)
        # 对话恒为单次请求，不接受压测式的自定义参数。
        self.assertEqual((job["requests"], job["concurrency"]), (1, 1))

    def test_create_chat_run_defaults_max_tokens(self):
        payload = json.loads(self._chat_payload())
        payload.pop("chat_max_tokens")
        payload.pop("chat_temperature")
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        job = put.call_args.args[0]
        self.assertEqual(job["chat_max_tokens"], server.CHAT_DEFAULT_MAX_TOKENS)
        self.assertIsNone(job["chat_temperature"])

    def test_chat_run_requires_prompt_and_valid_limits(self):
        for overrides in ({"chat_prompt": "   "}, {"chat_prompt": None},
                          {"chat_max_tokens": 0}, {"chat_max_tokens": 999999},
                          {"chat_max_tokens": "4096"}, {"chat_temperature": 9}):
            payload = json.loads(self._chat_payload())
            payload.update(overrides)
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/api/runs", data=json.dumps(payload).encode(), method="POST",
                                headers={"Content-Type": "application/json"}))
            self.assertEqual(caught.exception.code, 400, overrides)

    def test_chat_case_is_listed_with_text_families(self):
        cases = self.get_json("/api/cases")
        chat = [item for item in cases if item["test_kind"] == "chat"]
        self.assertEqual([item["id"] for item in chat], ["CHAT_PROMPT"])
        item = chat[0]
        self.assertEqual(item["protocols"], ["openai", "volcengine", "anthropic", "gemini"])
        self.assertTrue(item["fixed_parameters"])
        self.assertEqual(item["requests"], 1)
        self.assertEqual(item["executor"], "scripts/run_chat.py")
        self.assertIn("仅判定本次对话是否拿到正文", item["admission"])

    def test_chat_run_rejects_image_protocols(self):
        # openai_images / gemini_images 的家族与文字面相同，但 test_kinds 白名单不含 chat；
        # 若只看家族，Chat Completions 请求体会被发到 /images/generations。
        for protocol, model in (("openai_images", "gpt-image-2"), ("gemini_images", "gemini-3-pro-image")):
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/api/runs", data=self._chat_payload(protocol=protocol, model=model),
                                method="POST", headers={"Content-Type": "application/json"}))
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("不支持", json.loads(caught.exception.read())["error"])

    def test_chat_run_accepts_every_text_protocol(self):
        for protocol in ("openai", "openai_responses", "volcengine", "qianfan", "anthropic", "gemini"):
            with patch.object(server.JOB_QUEUE, "put") as put:
                with urlopen(Request(self.base + "/api/runs", data=self._chat_payload(protocol=protocol),
                                     method="POST", headers={"Content-Type": "application/json"})) as response:
                    result = json.load(response)
            self.assertEqual(result["test_kind"], "chat")
            self.assertEqual(put.call_args.args[0]["protocol"], protocol)

    def test_chat_child_dispatches_to_chat_runner_via_env(self):
        captured = self._execute_child_capture({
            "test_kind": "chat", "case": "CHAT_PROMPT", "protocol": "anthropic",
            "chat_prompt": "你好\n第二行", "chat_system": "你是主厨",
            "chat_max_tokens": 1024, "chat_temperature": 0.25,
        })
        self.assertEqual(Path(captured["command"][1]).name, "run_chat.py")
        self.assertEqual(captured["command"][2], "CHAT_PROMPT")
        # 提示词含换行，必须走环境变量而不是命令行。
        self.assertNotIn("你好\n第二行", [str(part) for part in captured["command"]])
        env = captured["env"]
        self.assertEqual(env["CHAT_PROMPT"], "你好\n第二行")
        self.assertEqual(env["CHAT_SYSTEM"], "你是主厨")
        self.assertEqual(env["CHAT_MAX_TOKENS"], "1024")
        self.assertEqual(env["CHAT_TEMPERATURE"], "0.25")
        self.assertEqual(env["CHAT_API_KEY"], "sk-x")
        self.assertEqual(env["CHAT_PROTOCOL"], "anthropic")
        self.assertEqual(env["CHAT_BASE_URL"], "https://i-token.ai/v1")

    def test_chat_response_endpoint_serves_raw_model_output(self):
        run_id = f"chat-response-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            result_dir = results / f"CHAT_PROMPT-20261012-130000-{run_id[:6]}"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text('{"admission_passed": true}', encoding="utf-8")
            (result_dir / "requests.jsonl").write_text('{"case": "CHAT_PROMPT"}\n', encoding="utf-8")
            (result_dir / "response.md").write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"/>", encoding="utf-8")
            with server.db() as connection:
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "chat", "CHAT_PROMPT", "openai", "https://mux.la", "gpt-5.6", "completed", 0,
                     time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), result_dir.name),
                )
            try:
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    detail = self.get_json(f"/api/runs/{run_id}")
                    with urlopen(f"{self.base}/api/runs/{run_id}/response") as response:
                        self.assertEqual(response.headers["Content-Type"], "text/markdown; charset=utf-8")
                        body = response.read().decode("utf-8")
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.assertEqual(detail["response_url"], f"/api/runs/{run_id}/response")
        self.assertEqual(body, '<svg xmlns="http://www.w3.org/2000/svg"/>')

    def test_plain_runs_have_no_response_url(self):
        run_id = f"no-response-{time.time_ns()}"
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, "load", "T0", "openai", "https://example.com", "m", "completed", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
            )
        try:
            self.assertIsNone(self.get_json(f"/api/runs/{run_id}")["response_url"])
        finally:
            with server.db() as connection:
                connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_gpt_pelican_artifacts_are_listed_and_directly_viewable(self):
        """鹈鹕骑车用例的产物：记录页暴露 artifacts 链接，/artifact/<name> 可直接在浏览器打开。"""
        run_id = f"pelican-artifacts-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            result_dir = results / f"GPT56_PELICAN-20261012-140500-{run_id[:6]}"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text('{"admission_passed": true}', encoding="utf-8")
            (result_dir / "requests.jsonl").write_text('{"case": "GPT56-62"}\n', encoding="utf-8")
            (result_dir / "pelican.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
            (result_dir / "pelican.html").write_text("<!doctype html><main><svg/></main>", encoding="utf-8")
            (result_dir / "pelican_response.md").write_text("画好了", encoding="utf-8")
            with server.db() as connection:
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "gpt", "GPT56_PELICAN", "openai", "https://gw", "gpt-5.6", "completed", 0,
                     time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), result_dir.name),
                )
            try:
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    detail = self.get_json(f"/api/runs/{run_id}")
                    self.assertEqual([item["name"] for item in detail["artifacts"]],
                                     ["pelican.html", "pelican.svg", "pelican_response.md"])
                    self.assertEqual(detail["artifacts"][0]["url"], f"/api/runs/{run_id}/artifact/pelican.html")
                    with urlopen(f"{self.base}/api/runs/{run_id}/artifact/pelican.html") as response:
                        self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")
                        self.assertIn("<svg", response.read().decode("utf-8"))
                    with urlopen(f"{self.base}/api/runs/{run_id}/artifact/pelican.svg") as response:
                        self.assertEqual(response.headers["Content-Type"], "image/svg+xml")
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_artifact_endpoint_only_serves_whitelisted_existing_files(self):
        """产物路由是白名单取值域：非白名单名、白名单里但未落盘、路径穿越都必须 404。"""
        run_id = f"pelican-guard-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            result_dir = results / f"GPT56_PELICAN-20261012-140600-{run_id[:6]}"
            result_dir.mkdir(parents=True)
            (result_dir / "pelican.html").write_text("<!doctype html><main/>", encoding="utf-8")
            (result_dir / "summary.json").write_text("{}", encoding="utf-8")
            with server.db() as connection:
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "gpt", "GPT56_PELICAN", "openai", "https://gw", "gpt-5.6", "completed", 0,
                     time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), result_dir.name),
                )
            try:
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    for name in ("summary.json", "..%2Fsummary.json", "pelican.svg"):
                        with self.assertRaises(HTTPError) as caught:
                            urlopen(f"{self.base}/api/runs/{run_id}/artifact/{name}")
                        self.assertEqual(caught.exception.code, 404, name)
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_plain_runs_have_no_artifacts(self):
        run_id = f"no-artifacts-{time.time_ns()}"
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, "load", "T0", "openai", "https://example.com", "m", "completed", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
            )
        try:
            self.assertEqual(self.get_json(f"/api/runs/{run_id}")["artifacts"], [])
        finally:
            with server.db() as connection:
                connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_gpt_pelican_cases_expose_multi_round_config(self):
        """产物类用例（鹈鹕骑车 SVG）在 /api/cases 里带轮数/并发上限：前端据此显示输入框。

        上限是**后端下发**的，前端不写死——改了 gpt56/ 的限制，界面随之变化，不用两头改。
        """
        from gpt56.cases import PELICAN_CASE_IDS
        from gpt56.core import PELICAN_MAX_CONCURRENCY, PELICAN_MAX_REQUESTS
        cases = {item["id"]: item for item in self.get_json("/api/cases")}
        self.assertTrue(PELICAN_CASE_IDS)
        for name in PELICAN_CASE_IDS:
            config = cases[name]["multi_round"]
            self.assertEqual(config["max_requests"], PELICAN_MAX_REQUESTS)
            self.assertEqual(config["max_concurrency"], PELICAN_MAX_CONCURRENCY)
            self.assertEqual((config["requests"], config["concurrency"]), (1, 1))
            self.assertIn("多轮", cases[name]["description"])
        # 其余 GPT 用例的请求数是写死的，不得带该字段（否则界面会冒出无意义的输入框）。
        others = [item for item in cases.values()
                  if item["test_kind"] == "gpt" and item["id"] not in PELICAN_CASE_IDS]
        self.assertTrue(others)
        self.assertFalse([item["id"] for item in others if "multi_round" in item])

    def test_gpt_pelican_rounds_are_validated_and_forwarded_to_cli(self):
        """轮数/并发：越界 → 400；合法 → 进 job，并在派发命令里以 --requests/--concurrency 出现。"""
        def payload(**overrides):
            body = {"protocol": "openai", "test_kind": "gpt", "base_url": "https://gw",
                    "api_key": "sk-test", "model": "gpt-5.6", "cases": ["GPT56_PELICAN"],
                    "requests": 5, "concurrency": 2}
            body.update(overrides)
            return json.dumps(body).encode()

        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload(), method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                self.assertEqual(response.status, 202)
        job = put.call_args[0][0]
        self.assertEqual((job["requests"], job["concurrency"]), (5, 2))
        self.assertEqual(server.pelican_round_args(job), ["--requests", "5", "--concurrency", "2"])

        for bad in (payload(requests=0), payload(requests=server.PELICAN_MAX_REQUESTS + 1),
                    payload(concurrency=0), payload(concurrency=server.PELICAN_MAX_CONCURRENCY + 1)):
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.base + "/api/runs", data=bad, method="POST",
                                headers={"Content-Type": "application/json"}))
            self.assertEqual(caught.exception.code, 400)

    def test_gpt_non_pelican_cases_drop_stale_rounds(self):
        """非产物类 GPT 用例：界面残留的轮数/并发必须被置空，且派发命令不带这两个参数。"""
        body = {"protocol": "openai", "test_kind": "gpt", "base_url": "https://gw",
                "api_key": "sk-test", "model": "gpt-5.6", "cases": ["GPT56_P2"],
                "requests": 7, "concurrency": 3}
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                self.assertEqual(response.status, 202)
        job = put.call_args[0][0]
        self.assertIsNone(job["requests"])
        self.assertIsNone(job["concurrency"])
        self.assertEqual(server.pelican_round_args(job), [])
        # 默认（界面未填）时也是 1×1，与 run_gpt56.py 的默认值一致。
        self.assertEqual(server.pelican_round_args({"test_kind": "gpt", "case": "GPT56_PELICAN"}),
                         ["--requests", "1", "--concurrency", "1"])

    def test_gpt_batch_applies_rounds_to_the_pelican_child_only(self):
        """同批提交「P0 + PELICAN」：轮数只落到产物类子任务，另一个子任务的请求数保持固定。

        回归点：判定若只看 ``case_names[0]``，首个用例不是产物类时整批的轮数会被丢弃，
        用户填的 3 轮会静默退化成 1 轮。
        """
        body = {"protocol": "openai", "test_kind": "gpt", "base_url": "https://gw",
                "api_key": "sk-test", "model": "gpt-5.6", "cases": ["GPT56_P0", "GPT56_PELICAN"],
                "requests": 3, "concurrency": 2}
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                self.assertEqual(response.status, 202)
        batch = put.call_args[0][0]
        self.assertTrue(batch["is_batch"])
        children = {child["case"]: child for child in batch["children"]}
        self.assertEqual((children["GPT56_PELICAN"]["requests"], children["GPT56_PELICAN"]["concurrency"]),
                         (3, 2))
        self.assertIsNone(children["GPT56_P0"]["requests"])
        self.assertIsNone(children["GPT56_P0"]["concurrency"])
        self.assertEqual(server.pelican_round_args(children["GPT56_PELICAN"]),
                         ["--requests", "3", "--concurrency", "2"])
        self.assertEqual(server.pelican_round_args(children["GPT56_P0"]), [])

    def test_pelican_round_svg_is_viewable_but_other_round_files_are_not(self):
        """多轮产物：`pelican-<轮次>.svg` 放行（总览页的相对链接要能直达），其余仍属未知产物。"""
        run_id = f"pelican-rounds-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            result_dir = results / f"GPT56_PELICAN-20261012-150000-{run_id[:6]}"
            result_dir.mkdir(parents=True)
            (result_dir / "pelican.html").write_text("<!doctype html><main/>", encoding="utf-8")
            (result_dir / "pelican-2.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
            (result_dir / "pelican-2.html").write_text("<!doctype html><main/>", encoding="utf-8")
            (result_dir / "summary.json").write_text("{}", encoding="utf-8")
            with server.db() as connection:
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "gpt", "GPT56_PELICAN", "openai", "https://gw", "gpt-5.6", "completed", 0,
                     time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), result_dir.name),
                )
            try:
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    with urlopen(f"{self.base}/api/runs/{run_id}/artifact/pelican-2.svg") as response:
                        self.assertEqual(response.headers["Content-Type"], "image/svg+xml")
                    # 逐轮**渲染页**不是产物（只有总览页是）：白名单只管文件取值域，不放开整目录。
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(f"{self.base}/api/runs/{run_id}/artifact/pelican-2.html")
                    self.assertEqual(caught.exception.code, 404)
                    # 记录行只列固定入口：200 轮也不该刷出 200 个链接。
                    detail = self.get_json(f"/api/runs/{run_id}")
                    self.assertEqual([item["name"] for item in detail["artifacts"]], ["pelican.html"])
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_official_kvv_child_uses_official_pytest_runner(self):
        # 同一个「Kimi 测试」模块里的官方套件仍走 run_kvv.py，且带位置参数（用例名）。
        captured = self._execute_child_capture({
            "test_kind": "kvv", "case": "KVV_PARAMS", "protocol": "volcengine",
            "temporary_case": None, "billing_probe": None,
        })
        self.assertEqual(Path(captured["command"][1]).name, "run_kvv.py")
        self.assertEqual(captured["command"][2], "KVV_PARAMS")
        self.assertNotIn("VOLC_TEMP_CASE", captured["env"])


if __name__ == "__main__": unittest.main()
