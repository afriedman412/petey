"""CLI behavior tests."""
import os


def test_main_loads_dotenv_from_cwd(tmp_path, monkeypatch, capsys):
    """`petey ...` invoked from a project dir must load that dir's
    .env, not one that lives near the petey source files. Regression
    guard for the editable-install gotcha where bare load_dotenv()
    walks up from cli.py's __file__.
    """
    var = "PETEY_TEST_DOTENV_FROM_CWD"
    (tmp_path / ".env").write_text(f"{var}=loaded-from-cwd\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("sys.argv", ["petey", "models", "path"])

    from petey.cli import main
    main()

    assert os.environ.get(var) == "loaded-from-cwd"
