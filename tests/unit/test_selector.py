import numpy as np
import pytest
from adaptive_self_assessment.selector import select_question, set_selector_seed

def test_get_question_reproducible():
    C = [f"item-{i}" for i in range(1, 16)]
    set_selector_seed(123)
    first = select_question(C)
    set_selector_seed(123)
    second = select_question(C)
    assert first == second

def test_select_question_empty_list():
    with pytest.raises(ValueError):
        select_question([])

def test_dynamic_question_selection():
    C = [f"item-{i}" for i in range(1, 16)]    
    set_selector_seed(np.random.randint(0,100))
    selected_items = []
    print("=== Dynamic Question Selection Test ===")
    while C:
        q = select_question(C)
        assert q in C
        assert q not in selected_items
        selected_items.append(str(q))
        C.remove(q)
        print (f"Done: {selected_items}")
        print(f"Remaining: {C}")
    assert len(selected_items) == 15


