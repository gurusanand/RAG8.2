"""Quick test to verify FAQ extraction produces COMPLETE, non-truncated answers."""
import sys
sys.path.insert(0, '.')

from rag_engine.faq_exact_match import FAQExactMatchEngine

# Test Q7 — short answer
test_text1 = """Q7. Which devices are compatible with Apple Pay?
All iPhone 6 models and above running iOS 11 are compatible for Apple Pay.

Q8. How do I add my Mashreq card to Apple Pay?
Open the Wallet app on your iPhone and tap the '+' sign to add a new card.
"""

# Test Q5 — medium answer
test_text2 = """Q5. How do I earn cashback on my SmartSaver Credit Card?
You will save on all purchases made on the SmartSaver Credit Card. Get 0.5% cash back on all local and 1.0% cash back on all international spends with no complicated calculations. Just pure savings! Cashback is valid on all spends including government and utility categories except cash transactions and Mashreq Online Banking payments.

Q6. What is the minimum spend to earn cashback?
There is no minimum spend required.
"""

# Test Q25 — LONG answer with two conditions (the truncation case)
test_text3 = """Q25. I am an existing Mashreq customer but do not transfer salary to Mashreq, will I be eligible for the benefits offered with the Mashreq Happiness Account?
If your existing Mashreq Account was opened after April 2016 and if you start to transfer a minimum salary of AED 10,000 then you will be eligible for all benefits of the Happiness Account including the joining bonus.If your existing Mashreq Account was opened prior to April 2016 and if you start to transfer a minimum salary of AED 10,000 then you will be eligible for all benefits of the Happiness Account excluding the joining bonus.

Q26. What is the next question?
Some answer here.
"""

class MockModel:
    def encode(self, texts, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            return np.random.randn(384)
        return np.random.randn(len(texts), 384)

class MockSettings:
    class rag:
        faq_exact_threshold = 0.85
        faq_fuzzy_threshold = 0.60
        faq_fuzzy_model = 'gpt-4.1-nano'

engine = FAQExactMatchEngine(MockModel(), MockSettings())

print("=" * 70)
print("TEST 1: Q7 Apple Pay (short answer)")
print("=" * 70)
pairs1 = engine.extract_faq_pairs(test_text1, "test.pdf")
for p in pairs1:
    print(f"\n{p.question_number}: {p.question}")
    print(f"ANSWER: {p.answer}")
    print(f"LENGTH: {len(p.answer)} chars")

print("\n" + "=" * 70)
print("TEST 2: Q5 SmartSaver Cashback (medium answer)")
print("=" * 70)
pairs2 = engine.extract_faq_pairs(test_text2, "test.pdf")
for p in pairs2:
    print(f"\n{p.question_number}: {p.question}")
    print(f"ANSWER: {p.answer}")
    print(f"LENGTH: {len(p.answer)} chars")

print("\n" + "=" * 70)
print("TEST 3: Q25 Happiness Account (LONG answer — must NOT be truncated)")
print("=" * 70)
pairs3 = engine.extract_faq_pairs(test_text3, "test.pdf")
for p in pairs3:
    print(f"\n{p.question_number}: {p.question}")
    print(f"ANSWER: {p.answer}")
    print(f"LENGTH: {len(p.answer)} chars")
    if p.question_number == "Q25":
        if "excluding the joining bonus" in p.answer:
            print("✅ PASS: Full answer with BOTH conditions preserved")
        else:
            print("❌ FAIL: Answer is TRUNCATED — missing second condition")
