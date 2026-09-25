"""Tests for src/target.py: target selection and SSH settings."""

import os
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from src import config, memory, target

_ALIAS = "tester@board-1"


def _native(**extra: str) -> dict:
    """Return the environment of a valid alias-mode native setup."""
    env = {
        "ATESOR_TARGET": "native",
        "ATESOR_PLATFORM": "debian",
        "ATESOR_SSH_HOST": _ALIAS,
    }
    env.update(extra)
    return env


def _field(**extra: str) -> dict:
    """Return the environment of a valid field-mode native setup."""
    env = {
        "ATESOR_TARGET": "native",
        "ATESOR_PLATFORM": "debian",
        "ATESOR_SSH_HOSTNAME": "board.example",
    }
    env.update(extra)
    return env


class TestLoadTarget(unittest.TestCase):
    """Tests for load_target and its validation rules."""

    def test_default_target_is_qemu(self) -> None:
        """No ATESOR_TARGET selects the qemu target."""
        cfg = target.load_target({})
        self.assertEqual(target.QEMU, cfg.name)
        self.assertTrue(cfg.is_valid)
        self.assertFalse(cfg.is_native)

    def test_qemu_ignores_native_keys(self) -> None:
        """SSH keys in .env do not break the qemu target."""
        cfg = target.load_target({"ATESOR_SSH_PASSWORD": "x"})
        self.assertTrue(cfg.is_valid)

    def test_invalid_target_is_reported(self) -> None:
        """An unknown ATESOR_TARGET is an error, not a silent qemu."""
        cfg = target.load_target({"ATESOR_TARGET": "arm"})
        self.assertFalse(cfg.is_valid)
        self.assertFalse(cfg.is_native)
        self.assertIn("qemu or native", cfg.errors[0])

    def test_alias_mode_is_valid(self) -> None:
        """ATESOR_SSH_HOST alone selects alias mode."""
        cfg = target.load_target(_native())
        self.assertTrue(cfg.is_valid, cfg.errors)
        self.assertTrue(cfg.is_native)
        self.assertEqual(target.ALIAS_MODE, cfg.mode)
        self.assertEqual(_ALIAS, cfg.ssh_host)
        self.assertEqual("~/atesor-ai", cfg.remote_workdir)

    def test_alias_accepts_ssh_destinations(self) -> None:
        """An alias, a host name, an address and user@host are valid."""
        for value in (
            "board",
            "tester@board-1",
            "10.0.0.5",
            "t_1@10.0.0.5",
            "my.host.example",
        ):
            with self.subTest(value=value):
                cfg = target.load_target(_native(ATESOR_SSH_HOST=value))
                self.assertTrue(cfg.is_valid, cfg.errors)

    def test_alias_rejects_unsafe_values(self) -> None:
        """Options, spaces and shell characters are rejected."""
        for value in (
            "-oProxyCommand=x",
            "a b",
            "a;b",
            "host:22",
            "a/b",
            "$(id)",
            "`id`",
            "u@h@x",
            "a\nb",
        ):
            with self.subTest(value=value):
                cfg = target.load_target(_native(ATESOR_SSH_HOST=value))
                self.assertFalse(cfg.is_valid)

    def test_platform_is_required_on_native(self) -> None:
        """Native needs alpine, debian or ubuntu, never auto."""
        env = _native()
        del env["ATESOR_PLATFORM"]
        self.assertFalse(target.load_target(env).is_valid)
        for value in ("auto", "fedora"):
            with self.subTest(value=value):
                cfg = target.load_target(_native(ATESOR_PLATFORM=value))
                self.assertFalse(cfg.is_valid)
        cfg = target.load_target(_native(ATESOR_PLATFORM="Ubuntu"))
        self.assertTrue(cfg.is_valid, cfg.errors)
        self.assertEqual("ubuntu", cfg.platform)

    def test_exactly_one_mode_is_required(self) -> None:
        """Neither mode and both modes are errors."""
        env = _native()
        del env["ATESOR_SSH_HOST"]
        neither = " ".join(target.load_target(env).errors)
        self.assertIn("in .env", neither)
        both = _native(ATESOR_SSH_HOSTNAME="board")
        self.assertIn("not both", " ".join(target.load_target(both).errors))

    def test_unknown_ssh_key_is_rejected(self) -> None:
        """A password key stops the start and its value is never shown."""
        cfg = target.load_target(_native(ATESOR_SSH_PASSWORD="hunter2"))
        self.assertFalse(cfg.is_valid)
        joined = " ".join(cfg.errors)
        self.assertIn("ATESOR_SSH_PASSWORD", joined)
        self.assertIn("key login only", joined)
        self.assertNotIn("hunter2", joined)

    def test_empty_ssh_values_are_ignored(self) -> None:
        """Empty placeholder keys count as unset."""
        cfg = target.load_target(
            _native(ATESOR_SSH_PASSWORD="", ATESOR_SSH_HOSTNAME="")
        )
        self.assertTrue(cfg.is_valid, cfg.errors)

    def test_field_mode_is_valid(self) -> None:
        """The field-mode keys give a field-mode configuration."""
        cfg = target.load_target(
            _field(
                ATESOR_SSH_PORT="2222",
                ATESOR_SSH_USER="tester",
                ATESOR_SSH_IDENTITY_FILE="~/.ssh/id_board",
            )
        )
        self.assertTrue(cfg.is_valid, cfg.errors)
        self.assertEqual(target.FIELD_MODE, cfg.mode)
        self.assertEqual(2222, cfg.port)
        self.assertEqual("tester", cfg.user)

    def test_field_mode_needs_hostname(self) -> None:
        """A field key without ATESOR_SSH_HOSTNAME is an error."""
        cfg = target.load_target(
            {
                "ATESOR_TARGET": "native",
                "ATESOR_PLATFORM": "debian",
                "ATESOR_SSH_USER": "tester",
            }
        )
        self.assertIn("needs ATESOR_SSH_HOSTNAME", " ".join(cfg.errors))

    def test_field_values_are_validated(self) -> None:
        """Values that could inject into the config file are rejected."""
        cases = {
            "ATESOR_SSH_HOSTNAME": ["-oProxyCommand=x", "a b", "a\nb"],
            "ATESOR_SSH_PORT": ["0", "65536", "abc", "²"],
            "ATESOR_SSH_USER": ["a b", "-x", "$(id)"],
            "ATESOR_SSH_IDENTITY_FILE": ["~/my key", "~/.ssh/%d", "~/'k'"],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    cfg = target.load_target(_field(**{key: value}))
                    self.assertFalse(cfg.is_valid)

    def test_workdir_characters_are_checked(self) -> None:
        """The work directory allows only a safe set of characters."""
        cfg = target.load_target(
            _native(ATESOR_REMOTE_WORKDIR="/data/atesor-ai")
        )
        self.assertTrue(cfg.is_valid, cfg.errors)
        for value in ("a b", "$(id)", "x;y", "~/it's"):
            with self.subTest(value=value):
                cfg = target.load_target(_native(ATESOR_REMOTE_WORKDIR=value))
                self.assertFalse(cfg.is_valid)


class TestTargetFlag(unittest.TestCase):
    """Tests for the early --target read that main.py runs."""

    def test_space_form_sets_environment(self) -> None:
        """--target native sets ATESOR_TARGET."""
        env: dict = {}
        target.apply_target_flag(["--repo", "u", "--target", "native"], env)
        self.assertEqual("native", env["ATESOR_TARGET"])

    def test_last_flag_wins(self) -> None:
        """The last --target wins, in both flag forms."""
        env = {"ATESOR_TARGET": "native"}
        target.apply_target_flag(["--target=native", "--target", "qemu"], env)
        self.assertEqual("qemu", env["ATESOR_TARGET"])

    def test_no_flag_keeps_environment(self) -> None:
        """Without the flag, the .env value stays."""
        env = {"ATESOR_TARGET": "native"}
        target.apply_target_flag(["--repo", "u"], env)
        self.assertEqual("native", env["ATESOR_TARGET"])

    def test_double_dash_ends_the_scan(self) -> None:
        """Arguments after -- are not options."""
        env: dict = {}
        target.apply_target_flag(["--", "--target", "native"], env)
        self.assertNotIn("ATESOR_TARGET", env)

    def test_target_name_from_env_is_tolerant(self) -> None:
        """Only native gives native; everything else gives qemu."""
        self.assertEqual(
            "native", target.target_name_from_env({"ATESOR_TARGET": "Native "})
        )
        self.assertEqual(
            "qemu", target.target_name_from_env({"ATESOR_TARGET": "arm"})
        )
        self.assertEqual("qemu", target.target_name_from_env({}))


class TestSshArgv(unittest.TestCase):
    """Tests for the ssh argv and the generated field-mode config."""

    def setUp(self) -> None:
        """Point the cache directory at a scratch folder."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_home = self._tmp.name

    def _env(self, base: dict) -> dict:
        """Return base plus the scratch cache and home directories."""
        env = dict(base)
        env["XDG_CACHE_HOME"] = self.cache_home
        env["HOME"] = self.cache_home
        return env

    def test_alias_mode_argv(self) -> None:
        """Alias mode passes the options and no config file."""
        with mock.patch.dict(os.environ, self._env(_native()), clear=True):
            argv = target.ssh_argv()
            alias = target.ssh_alias()
        self.assertEqual("ssh", argv[0])
        self.assertNotIn("-F", argv)
        for option in ("BatchMode=yes", "ControlMaster=auto"):
            self.assertIn(option, argv)
        cache = os.path.join(self.cache_home, "atesor-ai")
        self.assertIn(f"ControlPath={cache}/cm-{os.getpid()}-%C", argv)
        self.assertEqual(_ALIAS, alias)
        self.assertEqual(0o700, stat.S_IMODE(os.stat(cache).st_mode))

    def test_field_mode_writes_private_config(self) -> None:
        """Field mode writes the Host block with mode 0600."""
        env = self._env(
            _field(
                ATESOR_SSH_PORT="2222",
                ATESOR_SSH_USER="tester",
                ATESOR_SSH_IDENTITY_FILE="~/.ssh/id_board",
            )
        )
        with mock.patch.dict(os.environ, env, clear=True):
            argv = target.ssh_argv()
            alias = target.ssh_alias()
        path = argv[argv.index("-F") + 1]
        self.assertEqual("atesor-native", alias)
        self.assertEqual(0o600, stat.S_IMODE(os.stat(path).st_mode))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for line in (
            "Host atesor-native\n",
            "    HostName board.example\n",
            "    Port 2222\n",
            "    User tester\n",
            "    IdentityFile ~/.ssh/id_board\n",
            "    IdentitiesOnly yes\n",
            "    StrictHostKeyChecking accept-new\n",
        ):
            self.assertIn(line, text)

    def test_field_mode_keeps_the_address_out_of_argv(self) -> None:
        """Neither the host nor the user appears in the argv."""
        env = self._env(_field(ATESOR_SSH_USER="tester"))
        with mock.patch.dict(os.environ, env, clear=True):
            joined = " ".join(target.ssh_argv() + [target.ssh_alias()])
        self.assertNotIn("board.example", joined)
        self.assertNotIn("tester", joined)

    def test_minimal_field_config_omits_unset_lines(self) -> None:
        """Unset user and key lines are left out; the port defaults."""
        text = target.render_ssh_config(target.load_target(_field()))
        self.assertIn("    Port 22\n", text)
        self.assertNotIn("User ", text)
        self.assertNotIn("IdentityFile", text)
        self.assertNotIn("IdentitiesOnly", text)

    def test_native_helpers_need_a_valid_native_config(self) -> None:
        """The qemu target and a bad native config raise."""
        for env in ({}, {"ATESOR_TARGET": "native"}):
            with self.subTest(env=env):
                target.reset_target_cache()
                with mock.patch.dict(os.environ, self._env(env), clear=True):
                    with self.assertRaises(target.TargetConfigError):
                        target.ssh_argv()
                    with self.assertRaises(target.TargetConfigError):
                        target.ssh_alias()


class TestControlSocket(unittest.TestCase):
    """Tests for where the ssh control socket lives."""

    def setUp(self) -> None:
        """Create a scratch folder for the cache and runtime dirs."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_runtime_dir_holds_the_socket(self) -> None:
        """XDG_RUNTIME_DIR wins over the cache directory."""
        runtime = os.path.join(self._tmp.name, "run")
        os.mkdir(runtime)
        env = _native(XDG_RUNTIME_DIR=runtime, XDG_CACHE_HOME=self._tmp.name)
        with mock.patch.dict(os.environ, env, clear=True):
            argv = target.ssh_argv()
        socket_dir = os.path.join(runtime, "atesor-ai")
        self.assertIn(f"ControlPath={socket_dir}/cm-{os.getpid()}-%C", argv)
        self.assertEqual(0o700, stat.S_IMODE(os.stat(socket_dir).st_mode))

    def test_close_connection_closes_only_this_process(self) -> None:
        """Only sockets named with this process ID get an exit request."""
        socket_dir = os.path.join(self._tmp.name, "atesor-ai")
        os.mkdir(socket_dir)
        own = os.path.join(socket_dir, f"cm-{os.getpid()}-{'a' * 40}")
        other = os.path.join(socket_dir, f"cm-1-{'b' * 40}")
        for path in (own, other):
            open(path, "w").close()
        env = {"XDG_CACHE_HOME": self._tmp.name}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("src.target.subprocess.run") as run,
        ):
            run.return_value = mock.Mock(returncode=0, stderr="")
            target.close_connection()
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertEqual(["ssh", "-F", os.devnull], argv[:3])
        self.assertIn(f"ControlPath={own}", argv)
        self.assertEqual(["-O", "exit"], argv[-3:-1])
        self.assertIs(subprocess.DEVNULL, run.call_args.kwargs["stdin"])

    def test_close_connection_never_raises(self) -> None:
        """A failed close is logged, and the command still ends."""
        socket_dir = os.path.join(self._tmp.name, "atesor-ai")
        os.mkdir(socket_dir)
        open(os.path.join(socket_dir, f"cm-{os.getpid()}-x"), "w").close()
        env = {"XDG_CACHE_HOME": self._tmp.name}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "src.target.subprocess.run", side_effect=OSError("no ssh")
            ),
        ):
            target.close_connection()

    def test_close_connection_without_sockets_runs_nothing(self) -> None:
        """A process that never used ssh starts no ssh process."""
        env = {"XDG_CACHE_HOME": self._tmp.name}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("src.target.subprocess.run") as run,
        ):
            target.close_connection()
        run.assert_not_called()

    def test_missing_runtime_dir_falls_back_to_the_cache(self) -> None:
        """A runtime dir that does not exist is not used."""
        env = {
            "XDG_RUNTIME_DIR": os.path.join(self._tmp.name, "gone"),
            "XDG_CACHE_HOME": self._tmp.name,
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                os.path.join(self._tmp.name, "atesor-ai"),
                target.control_dir(),
            )

    def test_long_socket_path_does_not_fit(self) -> None:
        """A path over the Unix socket limit is reported as too long."""
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/c"}, clear=True):
            self.assertTrue(target.control_path_fits())
        with mock.patch.dict(
            os.environ, {"XDG_CACHE_HOME": "/c" + "x" * 60}, clear=True
        ):
            self.assertFalse(target.control_path_fits())


class TestRemoteHelpers(unittest.TestCase):
    """Tests for the remote path, workdir and redaction helpers."""

    def test_remote_path_expr_expands_home(self) -> None:
        """A leading ~ becomes $HOME; the rest is one quoted word."""
        self.assertEqual('"$HOME"', target.remote_path_expr("~"))
        self.assertEqual(
            '"$HOME"/atesor-ai', target.remote_path_expr("~/atesor-ai")
        )
        self.assertEqual(
            "/data/atesor-ai", target.remote_path_expr("/data/atesor-ai")
        )

    def test_resolved_workdir_resets_with_the_cache(self) -> None:
        """The resolved work directory is per run."""
        target.set_remote_workdir("/home/t/atesor-ai")
        self.assertEqual("/home/t/atesor-ai", target.remote_workdir())
        target.reset_target_cache()
        self.assertIsNone(target.remote_workdir())

    def test_redact_hides_field_mode_details(self) -> None:
        """Field-mode host, user and any IPv4 address are hidden."""
        env = _field(ATESOR_SSH_USER="tester")
        with mock.patch.dict(os.environ, env, clear=True):
            text = target.redact(
                "connect to board.example as tester via 10.1.2.3"
            )
        self.assertNotIn("board.example", text)
        self.assertNotIn("tester", text)
        self.assertNotIn("10.1.2.3", text)
        self.assertIn("<ATESOR_SSH_HOSTNAME>", text)


class TestMachineLabel(unittest.TestCase):
    """Tests for the machine name in console messages."""

    def _label(self, env: dict, name: str = "") -> str:
        """Return the machine label for an environment and a host name."""
        with mock.patch.dict(os.environ, env, clear=True):
            target.reset_target_cache()
            if name:
                target.set_machine_name(name)
            return target.machine_label()

    def test_alias_mode_names_the_destination_and_host(self) -> None:
        """Alias mode shows the destination, and a host name that differs."""
        self.assertEqual(_ALIAS, self._label(_native()))
        self.assertEqual(_ALIAS, self._label(_native(), "board-1"))
        self.assertEqual(
            f"{_ALIAS} (host board-os)", self._label(_native(), "board-os")
        )

    def test_field_mode_keeps_the_address_private(self) -> None:
        """Field mode shows only the host name, through redact()."""
        self.assertEqual("the native machine", self._label(_field()))
        self.assertEqual("host board-os", self._label(_field(), "board-os"))
        self.assertEqual(
            "host <ATESOR_SSH_HOSTNAME>",
            self._label(_field(), "board.example"),
        )

    def test_reset_forgets_the_host_name(self) -> None:
        """A new run does not show the host name of an old run."""
        with mock.patch.dict(os.environ, _native(), clear=True):
            target.reset_target_cache()
            target.set_machine_name("board-os")
            target.reset_target_cache()
            self.assertEqual(_ALIAS, target.machine_label())


class TestTargetSeparation(unittest.TestCase):
    """Native and qemu results never share a workspace or cache key."""

    def test_native_workspace_is_separate(self) -> None:
        """The native target uses workspace-native under the state home."""
        if config.is_running_in_docker():
            self.skipTest("running inside docker")
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"ATESOR_HOME": home}):
                os.environ.pop("ATESOR_TARGET", None)
                self.assertEqual(
                    os.path.join(home, "workspace"),
                    config.get_workspace_root(),
                )
                os.environ["ATESOR_TARGET"] = "native"
                self.assertEqual(
                    os.path.join(home, "workspace-native"),
                    config.get_workspace_root(),
                )

    def test_recipe_cache_key_has_native_suffix(self) -> None:
        """Only the native target adds -native to the sandbox key."""
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("ATESOR_TARGET", None)
            self.assertEqual("alpine-riscv64", memory._default_sandbox())
            os.environ["ATESOR_TARGET"] = "native"
            target.reset_target_cache()
            self.assertEqual(
                "alpine-riscv64-native", memory._default_sandbox()
            )


if __name__ == "__main__":
    unittest.main()
