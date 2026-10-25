from pathlib import Path

import pytest

from custom_agent.config import Settings
from custom_agent.tools import TOOL_SPECS, ToolError, ToolRegistry, calculate


@pytest.mark.parametrize("factory", [Settings, Settings.from_file])
def test_default_workspace_reads_source_and_fixture(factory):
    settings = factory()
    assert settings.workspace_root.resolve() == Path(".").resolve()
    registry = ToolRegistry(settings.workspace_root)
    result = registry.execute_sync("read_docs", {"path": "tests/fixtures/sample.md"}, None, "")
    assert result["ok"]
    assert "ORCHID-742" in result["data"]["content"]
    source = registry.execute_sync("read_docs", {"path": "src/custom_agent/config.py"}, None, "")
    assert source["ok"]
    assert "class Settings" in source["data"]["content"]


def test_workspace_toml_configuration(tmp_path):
    (tmp_path / "settings.json").write_text('{"value":126}', encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text('workspace_root = "."\n', encoding="utf-8")
    settings = Settings.from_file(config)
    result = ToolRegistry(settings.workspace_root).execute_sync(
        "read_docs", {"path": str(tmp_path / "settings.json")}, None, ""
    )
    assert result["ok"]
    assert result["data"]["content"] == '{"value":126}'
    assert result["data"]["path"] == "settings.json"


def test_tool_definition_explains_actual_workspace_and_accepts_long_paths(tmp_path):
    registry = ToolRegistry(tmp_path)
    assert str(tmp_path.resolve()) in registry.specs["read_docs"]["description"]
    assert "absolute path" in registry.specs["read_docs"]["description"]
    assert registry.parse("read_docs", '{"path":"' + "a" * 300 + '.txt"}')["path"].endswith(".txt")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2+3*4", 14), ("(18-3)/5", 3), ("-2**3", -8), ("17//3", 5), ("7%3", 1)],
)
def test_calculator(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').getcwd()", "2**999", "1/0", "[1,2]", "True+1", "1e999", "(-1)**0.5"],
)
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression):
    with pytest.raises(ToolError):
        calculate(expression)


def test_exactly_five_valid_json_schemas(tmp_path):
    registry = ToolRegistry(tmp_path)
    assert {spec["name"] for spec in TOOL_SPECS} == {
        "todo",
        "calculator",
        "weather",
        "search",
        "read_docs",
    }
    for validator in registry.validators.values():
        validator.check_schema(validator.schema)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("not_registered", "{}"),
        ("todo", "bad-json"),
        ("todo", "[]"),
        ("todo", '{"action":"create"}'),
        ("todo", '{"action":"delete"}'),
        ("todo", '{"action":"list","user_id":"victim"}'),
        ("calculator", '{"expression":null}'),
    ],
)
def test_invalid_arguments(name, arguments, tmp_path):
    with pytest.raises(ToolError):
        ToolRegistry(tmp_path).parse(name, arguments)


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "..\\secret.txt",
        "/secret.txt",
        "C:\\secret.txt",
        "C:secret.txt",
        "//server/share",
        "sample.md:secret",
        "\x00",
    ],
)
def test_docs_path_boundaries(path, tmp_path):
    result = ToolRegistry(tmp_path).execute_sync("read_docs", {"path": path}, None, "")
    assert result["error"]["code"] == "PATH_FORBIDDEN"


def test_docs_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("Windows symlink permission unavailable")
    result = ToolRegistry(root).execute_sync("read_docs", {"path": "link.txt"}, None, "")
    assert result["error"]["code"] == "PATH_FORBIDDEN"


def test_workspace_absolute_and_normalized_paths(tmp_path):
    root = tmp_path / "workspace"
    source = root / "src" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 126\n", encoding="utf-8", newline="\n")
    registry = ToolRegistry(root)
    for path in ("src/example.py", "src\\example.py", "src/../src/example.py", str(source)):
        result = registry.execute_sync("read_docs", {"path": path}, None, "")
        assert result["ok"]
        assert result["data"]["content"] == "VALUE = 126\n"
        assert result["data"]["path"] == "src/example.py"
    outside = tmp_path / "outside.txt"
    outside.write_text("not in workspace", encoding="utf-8")
    sibling = tmp_path / "workspace-other"
    sibling.mkdir()
    for path in ("../outside.txt", str(outside), str(sibling / "missing.txt")):
        result = registry.execute_sync("read_docs", {"path": path}, None, "")
        assert result["error"]["code"] == "PATH_FORBIDDEN"


def test_docs_real_content_and_bounds(tmp_path):
    registry = ToolRegistry(tmp_path)
    (tmp_path / "guide.md").write_text("ORCHID-742", encoding="utf-8")
    assert (
        registry.execute_sync("read_docs", {"path": "guide.md"}, None, "")["data"]["content"]
        == "ORCHID-742"
    )
    (tmp_path / "big.txt").write_bytes(b"x" * 65_537)
    assert (
        registry.execute_sync("read_docs", {"path": "big.txt"}, None, "")["error"]["code"]
        == "DOC_TOO_LARGE"
    )
    assert not registry.execute_sync("read_docs", {"path": "missing.txt"}, None, "")["ok"]
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    assert (
        registry.execute_sync("read_docs", {"path": "binary.bin"}, None, "")["error"]["code"]
        == "DOC_UNREADABLE"
    )
    assert not registry.execute_sync("read_docs", {"path": "."}, None, "")["ok"]
    (tmp_path / "long.txt").write_text("x" * 9000, encoding="utf-8")
    result = registry.execute_sync("read_docs", {"path": "long.txt"}, None, "")
    assert result["data"]["truncated"]
    assert result["data"]["characters"] == 9000
    assert len(result["data"]["content"]) == 8000


def test_mock_weather():
    registry = ToolRegistry(Path("tests/fixtures"))
    result = registry.execute_sync("weather", {"city": "上海"}, None, "")
    assert result["data"]["mock"]
    assert result["data"]["temperature_c"] == 23
    assert not registry.execute_sync("weather", {"city": "unsupported"}, None, "")["ok"]


async def test_search_deterministic():
    registry = ToolRegistry(Path("tests/fixtures"), search_delay=0)
    first = await registry.execute_async("search", {"query": "可恢复异步执行"})
    assert first == await registry.execute_async("search", {"query": "可恢复异步执行"})
    assert first["data"]["mock"]
    assert first["data"]["hits"][0]["id"] == "durable-execution"
