import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

class ReportUiTests(unittest.TestCase):
    def test_markdown_renderer_and_styles_exist(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        self.assertIn('function renderMarkdown',js)
        self.assertIn("kind==='report'",js)
        self.assertIn('.markdown-body h1',css)
        self.assertIn('.md-table-wrap',css)

    def test_batch_report_renders_expandable_suites_with_metrics(self):
        """批量聚合报告：指标列（正式请求/成功率/TTFT/TTFC/E2E）+ 套件行可展开用例明细。"""
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        server=(ROOT/'web/server.py').read_text(encoding='utf-8')
        self.assertIn('function renderBatchReport',js)
        self.assertIn('function batchSuiteTable',js)
        self.assertIn('caseBadge',js)
        for column in ('请求','成功率','用例通过','TTFT P90','TTFC P90','E2E P90','TPOT(ms/tok)'):
            self.assertIn(column,js)
        # 交互：点击套件行切换展开态（事件委托 + hidden 切换）
        self.assertIn(".closest('.suite-row')",js)
        self.assertIn('suite-detail',js)
        # 表格不横向滚动：无吸底条，子记录 ID 收进展开明细，表头紧凑
        self.assertNotIn('batch-hscroll',js)
        self.assertNotIn('wireBatchHScroll',js)
        self.assertNotIn('batch-hscroll',css)
        self.assertIn('case-record',js)
        self.assertIn('<th>TPOT(ms/tok)</th>',js)
        self.assertIn('.batch-suites th,.batch-suites td{padding:7px 10px}',css)
        # 样式与徽标
        self.assertIn('.suite-row.open',css)
        self.assertIn('.case-badge.passed',css)
        self.assertIn('.case-badge.failed',css)
        # 旧批次：manifest 读取时即时补齐 metrics
        self.assertIn('ensure_batch_metrics',server)

    def test_case_details_show_complete_execution_parameters(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        self.assertIn("caseFacts(item)", js)
        for label in ("并发 / 速率", "输入 Token", "总请求数", "预热时间", "稳定运行"):
            self.assertIn(label, js)
        # 原本的信息必须回到卡片上：来源·执行器一行 + 执行参数 dl + 准入标准。
        self.assertIn('<dl class="case-facts">', js)
        self.assertIn('item.source', js)
        self.assertIn('item.executor', js)
        self.assertIn('准入标准', js)
        self.assertIn('admission-standard', js)
        self.assertIn('admission-standard', css)
        self.assertIn('case-facts', css)

    def test_test_workspace_has_list_and_detail_panes_with_existing_controls(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        for label in ('测试项列表', '测试项详情与配置', '本次测试模型'):
            self.assertIn(label, html)
        for element in ('caseGrid', 'caseDetail', 'caseEvidence', 'testExecutionConfigs', 'casePresets',
                        'clearCaseSelection', 'casePresetStatus', 'model',
                        'customConcurrency', 'chatPrompt', 'imagePrompt', 'seedancePrompt'):
            self.assertEqual(html.count(f'id="{element}"'), 1)
        self.assertIn('class="test-case-button"', js)
        self.assertNotIn('class="test-case-check"', js)
        self.assertIn('class="test-case-detail-button"', js)
        self.assertIn('<script src="/case-presets.js"></script>', html)
        self.assertNotIn('class="case-card', js)
        self.assertIn('caseScopeOptions(item)', js)
        self.assertIn('class="case-face-toggle"', js)
        self.assertIn('function renderTestModels', js)
        self.assertIn('grid-template-columns:290px minmax(0,1fr)', css)

    def test_generated_reports_include_ttft(self):
        reporting=(ROOT/'volcperf/reporting.py').read_text(encoding='utf-8')
        aggregation=(ROOT/'web/aggregation.py').read_text(encoding='utf-8')
        self.assertIn('TTFT P50/P90/P99',reporting)
        self.assertIn('ttft_p90_seconds',reporting)
        self.assertIn('TTFT P90(s)',aggregation)

    def test_non_report_artifacts_stay_plain_text(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        self.assertIn("className='report-body code-body'",js)
        self.assertIn("textContent=text",js)

    def test_model_uses_searchable_combobox(self):
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        self.assertIn('id="modelSuggestions"',html)
        self.assertNotIn('id="modelSelect"',html)
        self.assertIn('function fuzzyModelMatch',js)
        self.assertIn("$('#model').oninput",js)
        self.assertIn('role="combobox"',html)

    def test_key_is_retained_and_password_save_is_discouraged(self):
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        self.assertIn('id="apiKey"',html)
        self.assertIn('type="text" class="secret-masked"',html)
        self.assertIn('autocomplete="off"',html)
        self.assertIn('data-lpignore="true"',html)
        self.assertIn('data-1p-ignore="true"',html)
        self.assertNotIn("$('#apiKey').value=''",js)

    def test_chat_panel_is_wired_end_to_end(self):
        """对话窗口：面板 DOM、渲染函数与 DOM id 一致性。

        前端最容易漏的是「JS 里引用了 id、index.html 里没有」——这类错误在浏览器里
        只在点击时才炸，静态断言可以提前拦住。
        """
        import re
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        for element_id in ('chatConfig','chatPrompt','chatSystem','chatMaxTokens',
                           'chatTemperature','chatSend','chatCopy','chatRaw','chatStatus','chatResult'):
            self.assertIn(f'id="{element_id}"',html,f'index.html 缺少 #{element_id}')
        for function in ('function renderChatConfig','function pollChatRun','function showChatResult'):
            self.assertIn(function,js)
        self.assertIn("state.testKind==='chat'",js)
        self.assertIn('chat_prompt:',js)
        self.assertIn(".chat-panel",css)
        self.assertIn(".chat-result",css)
        # JS 里出现的每个 #chatX 选择器都必须在 index.html 中存在。
        referenced=set(re.findall(r"\$\('#(chat[A-Za-z]+)'\)",js))
        self.assertTrue(referenced)
        self.assertEqual({x for x in referenced if f'id="{x}"' not in html}, set())

    def test_generated_artifacts_are_linkable_from_report_and_records(self):
        """产物「可直接查看」的两个入口都要在：报告内的站内链接 + 记录行的新标签页链接。

        报告由 ``renderSingleReport`` → ``renderMarkdown`` 渲染，而渲染器原先只把
        ``https?://`` 认成链接；报告的产物链接是站内相对地址（``/api/runs/<id>/artifact/...``），
        不放宽这一条，报告里就只剩一串死文本——「可直接查看」等于没做到。
        """
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        # 站内相对链接必须被渲染成 <a>。
        self.assertIn(r"(?:https?:\/\/|\/)", js)
        self.assertIn('function artifactLinks', js)
        self.assertIn('function artifactLabel', js)
        self.assertIn('run.artifacts', js)
        # 记录行的产物入口是新标签页直开（html 看渲染视图、svg 看原图）。
        self.assertIn('target="_blank"', js)
        self.assertIn('artifact-btn', css)
        # 报告「产物」小节的链接目标由服务端生成，格式与前端断言保持一致。
        core=(ROOT/'gpt56/core.py').read_text(encoding='utf-8')
        self.assertIn('/api/runs/{record_id}/artifact/{name}', core)

    def test_pelican_multi_round_panel_is_wired(self):
        """鹈鹕骑车用例的「请求轮数 / 并发数」输入框：DOM、取值、提交与显隐都要接上。

        前端最典型的漏接线是"输入框存在但提交时不带"（参数被静默忽略，界面看起来正常），
        所以这里同时钉死：元素存在、提交体里带键、显隐由 multi_round 字段驱动。
        """
        import re
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        for element_id in ('gptPelicanConfig','gptRequests','gptConcurrency'):
            self.assertIn(f'id="{element_id}"',html,f'index.html 缺少 #{element_id}')
        self.assertIn('function renderPelicanConfig',js)
        self.assertIn('multi_round',js)
        # 提交体必须真的把这两个值发出去（否则后端只会拿到 None，退化成单轮）。
        self.assertIn('pelicanRounds',js)
        self.assertIn('pelicanConcurrency',js)
        self.assertIn('requests:',js)
        self.assertIn('concurrency:',js)
        self.assertIn(".pelican-config",css)
        # JS 里出现的每个 #gptX 选择器都必须在 index.html 中存在。
        referenced=set(re.findall(r"\$\('#(gpt[A-Za-z]+)'\)",js))
        self.assertTrue(referenced)
        self.assertEqual({x for x in referenced if f'id="{x}"' not in html}, set())

    def test_gpt_image_hold_duration_panel_is_wired(self):
        """图片压测「压测持续时长」：勾选框 + 秒数框都要接上，且**默认不启用**。

        两个典型漏接线：控件存在但提交时不带（参数被静默忽略）；或没勾选也照发
        （把单批被动变成时长压测——历史事故：这里的「请求时间」被当成单请求超时填成 1 秒，
        整批请求在第 1.002 秒被本地掐断）。因此同时钉死：元素存在、默认不勾选、
        秒数框随勾选启停、提交体只在勾选时带键。
        """
        import re
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        for element_id in ('imageHoldField','imageHoldEnabled','imageHoldSeconds'):
            self.assertIn(f'id="{element_id}"',html,f'index.html 缺少 #{element_id}')
        # 默认不勾选，且秒数框初始禁用（不启用时不可编辑）。
        self.assertIn('id="imageHoldEnabled" type="checkbox"',html)
        self.assertNotIn('id="imageHoldEnabled" type="checkbox" checked',html)
        self.assertIn('id="imageHoldSeconds" type="number" min="1" step="1" value="60" disabled',html)
        # 旧字段彻底退场：不能再把「请求时间」当单请求超时提交。
        self.assertNotIn('imageRequestTimeout',js)
        self.assertNotIn('image_request_timeout_seconds',js)
        # 提交体只在勾选时才带压测持续时长。
        self.assertIn('image_hold_seconds',js)
        self.assertIn("$('#imageHoldEnabled').checked",js)
        self.assertIn('function syncImageHoldState',js)
        # 「恢复默认」要把勾选与数值一起清掉，否则用户改过一次就再也回不到默认。
        self.assertIn("delete $('#imageHoldEnabled').dataset.edited",js)
        self.assertIn("delete $('#imageHoldSeconds').dataset.edited",js)
        referenced=set(re.findall(r"\$\('#(imageHold[A-Za-z]+)'\)",js))
        self.assertTrue(referenced)
        self.assertEqual({x for x in referenced if f'id="{x}"' not in html}, set())

if __name__=='__main__':unittest.main()
