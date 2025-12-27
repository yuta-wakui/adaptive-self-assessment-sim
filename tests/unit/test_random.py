import numpy as np
import pytest
from adaptive_self_assessment.random import make_selector_seed

def test_make_selector_seed_reproducible():
    s1 = make_selector_seed(42, 3, "userA")
    s2 = make_selector_seed(42, 3, "userA")
    assert s1 == s2

def test_make_selector_seed_different_users():
    s1 = make_selector_seed(42, 3, "userA")
    s2 = make_selector_seed(42, 3, "userB")
    assert s1 != s2