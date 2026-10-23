import unittest
from volcperf.data import PromptRecord
from volcperf.prompts import render_workload
from volcperf.protocols import payload


class WorkloadTests(unittest.TestCase):
    def test_five_business_workload_types(self):
        workloads=[render_workload("m",200,0,index) for index in range(5)]
        self.assertEqual([w.workload_type for w in workloads],["qa","summary","code","json","multi_turn"])
        self.assertEqual(len(workloads[-1].messages),3)
        self.assertEqual(workloads[-1].messages[1]["role"],"assistant")
        self.assertTrue(all(w.target_visible_tokens < 512 for w in workloads))
        self.assertNotIn("until the configured output limit", " ".join(m["content"] for w in workloads for m in w.messages))

    def test_record_variant_is_deterministic(self):
        record=PromptRecord("I1-00004","I1",7500,"abc",100,0)
        self.assertEqual(record.workload().workload_type,"multi_turn")

    def test_protocol_preserves_multi_turn_and_provider_default_thinking(self):
        messages=[{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"u2"}]
        body=payload("openai","m",messages,256,True)
        self.assertEqual([x["role"] for x in body["messages"]],["system","user","assistant","user"])
        self.assertNotIn("reasoning_effort",body)
        forced=payload("openai","m",messages,256,True,"high")
        self.assertEqual(forced["reasoning_effort"],"high")
        anthropic=payload("anthropic","m",messages,256,True)
        self.assertEqual(len(anthropic["messages"]),3)
        gemini=payload("gemini","m",messages,256,True)
        self.assertEqual([x["role"] for x in gemini["contents"]],["user","model","user"])

if __name__=="__main__": unittest.main()
