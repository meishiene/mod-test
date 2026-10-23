import unittest
from volcperf.cases import CASES
from volcperf.runner import load_case_prompts

class B1Tests(unittest.TestCase):
    def test_combined_b1_distribution_and_concurrency(self):
        case=CASES["B1"]
        prompts=load_case_prompts(case)
        self.assertEqual(case.concurrency,2)
        self.assertEqual(case.samples,16)
        self.assertEqual(len(prompts),16)
        counts={}
        for prompt in prompts: counts[prompt.dataset]=counts.get(prompt.dataset,0)+1
        self.assertEqual(counts,{"I1":4,"I2":4,"I3":4,"I4":4})

if __name__ == "__main__": unittest.main()
