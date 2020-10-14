
from model_tests import *
print("Test 1 (Vanilla): ", "[OK]" if test_1() else "[Fail]")
print("Test 2 (L2): ", "[OK]" if test_2() else "[Fail]")
print("Test 3 (L2+FFT inp): ", "[OK]" if test_3() else "[Fail]")
print("Test 4 (ParallelSum): ", "[OK]" if test_4() else "[Fail]")
print("Test 5 (FIR): ", "[OK]" if test_5() else "[Fail]")
print("Test 6 (FIR Multi): ", "[OK]" if test_6() else "[Fail]")
print("Test 7 (FIR Multi + L2 + HF supp): ", "[OK]" if test_7() else "[Fail]")