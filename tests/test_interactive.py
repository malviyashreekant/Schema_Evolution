import sys
from io import StringIO

from rich.console import Console

from schema_evolution import cli, interactive


class ExitPrompt:
    def ask(self):
        return "exit"


def test_interactive_menu_can_exit_cleanly(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(interactive, "console", Console(file=output, force_terminal=False))
    monkeypatch.setattr(interactive.questionary, "select", lambda *args, **kwargs: ExitPrompt())

    interactive.run_interactive()

    assert "SCHEMA EVOLUTION" in output.getvalue()
    assert "Goodbye." in output.getvalue()


def test_cli_without_subcommand_launches_interactive_menu(monkeypatch):
    launched = []
    monkeypatch.setattr(sys, "argv", ["schema-evolution"])
    monkeypatch.setattr(interactive, "run_interactive", lambda: launched.append(True))

    cli.main()

    assert launched == [True]