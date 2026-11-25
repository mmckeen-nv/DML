"""Generated overlays carry current preferences, not a bundled persona name."""
from types import SimpleNamespace
from daystrom_dml.personality_matrix import PersonalityMatrix


def matrix(tmp_path):
    return PersonalityMatrix(SimpleNamespace(enable=True, mode="active-read", max_overlay_chars=4000), storage_dir=tmp_path)


def test_structured_overlay_does_not_inject_identity(tmp_path):
    rendered = matrix(tmp_path)._structured_overlay_text(["Use uppercase headings."], ["Respect the current request."])
    assert rendered.startswith("Preferences:")
    assert "Identity:" not in rendered
    assert "Use uppercase headings." in rendered


def test_evolution_overlay_does_not_inject_identity(tmp_path):
    service = matrix(tmp_path)
    graph = service._load_or_create_evolution_graph()
    overlay = service._overlay_from_evolution_graph(graph, prompt="Explain this function", thread_id=None, project_id=None, relationship_id=None)
    assert overlay is not None
    rendered = overlay["overlay"]["rendered_text"]
    assert rendered.startswith("Current tendency:")
    assert "Identity:" not in rendered
