from llmbench.models import is_embedding_model, param_sort_key


def m(name: str, **details: object) -> dict:
    return {"name": name, "details": details}


def test_param_sort_key_parses_common_sizes():
    assert param_sort_key(m("a", parameter_size="27.8B")) == 27.8e9
    assert param_sort_key(m("a", parameter_size="4B")) == 4e9
    assert param_sort_key(m("a", parameter_size="500M")) == 500e6


def test_param_sort_key_handles_missing_and_garbage():
    assert param_sort_key(m("a")) == 0.0
    assert param_sort_key(m("a", parameter_size="unknown")) == 0.0
    assert param_sort_key({"name": "a"}) == 0.0


def test_sorting_is_ascending_by_size():
    models = [
        m("big", parameter_size="70B"),
        m("small", parameter_size="4B"),
        m("mid", parameter_size="27B"),
    ]
    assert [x["name"] for x in sorted(models, key=param_sort_key)] == ["small", "mid", "big"]


def test_embedding_models_detected_by_name_and_family():
    assert is_embedding_model(m("nomic-embed-text"))
    assert is_embedding_model(m("something", families=["bert-embedding"]))
    assert not is_embedding_model(m("qwen3.5:27b", families=["qwen3"]))
    assert not is_embedding_model(m("llama3.1:8b"))
