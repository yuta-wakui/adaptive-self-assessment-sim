from adaptive_self_assessment.components.model_store import ModelStore

class DummyPipeline:
    """A lightweight stand-in for sklearn.pipeline.Pipeline in unit tests."""
    pass

def test_model_store_get_returns_none_when_missing():
    store = ModelStore()
    key = ("ws1", 0, "item_1")
    assert store.get(key) is None

def test_model_store_set_then_get_returns_same_object():
    store = ModelStore()
    key = ("ws1", 0, "item_1")
    model = DummyPipeline()

    store.set(key, model)  # type: ignore[arg-type]
    assert store.get(key) is model

def test_model_store_overwrites_existing_key():
    store = ModelStore()
    key = ("ws1", 0, "item_1")
    model1 = DummyPipeline()
    model2 = DummyPipeline()

    store.set(key, model1)  # type: ignore[arg-type]
    store.set(key, model2)  # type: ignore[arg-type]
    assert store.get(key) is model2