"""Tests for src/machine.py: the machine probe and the flag rules."""

import ast
import json
import unittest
from unittest import mock

from src import machine
from src.machine import (
    MachineProfile,
    align_flags,
    describe_rules,
    evaluate_probe,
    forbidden_classes,
    goriscv64_for,
    hwprobe_extensions,
    march_for,
    match_core,
    misaligned_from_timing,
    misaligned_speed,
    parse_cpuinfo,
    parse_isa_string,
    probe_machine,
    profile_label,
    qemu_cpu_for,
    rust_flags_for,
    tune_for,
)
from src.state import CommandResult

# The ISA, uarch and ids match the live probe of the Megrez (2026-10-01).
MEGREZ_ISA = "rv64imafdch_zicntr_zicsr_zifencei_zihpm_zba_zbb_sscofpmf"
MEGREZ_CPUINFO = (
    "processor\t: 0\n"
    "hart\t\t: 0\n"
    f"isa\t\t: {MEGREZ_ISA}\n"
    "mmu\t\t: sv48\n"
    "uarch\t\t: eswin,eic770x\n"
    "mvendorid\t: 0x489\n"
    "marchid\t\t: 0x8000000000000008\n"
    "mimpid\t\t: 0x6220425\n"
    "\n"
    "processor\t: 1\n"
    "isa\t\t: rv64imafdc_zbs\n"
)
X86_CPUINFO = (
    "processor\t: 0\n"
    "vendor_id\t: GenuineIntel\n"
    "flags\t\t: fpu vme sse sse2\n"
)
QEMU_MAX_CPUINFO = (
    "processor\t: 0\n"
    "hart\t\t: 0\n"
    "isa\t\t: rv64imafdcvh_zicond_zfa_zfhmin_zba_zbb_zbs\n"
    "mmu\t\t: sv48\n"
    "uarch\t\t: qemu\n"
)
# BASE_BEHAVIOR_IMA, and IMA_EXT_0 = FD | C | ZBA | ZBB.
MEGREZ_PAIRS = [
    [0, 0x489],
    [1, 0x8000000000000008],
    [2, 0x6220425],
    [3, 1],
    [4, (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)],
    [5, 2],
]
SIGILL = -4


def _tests(present=(), absent=(), error=()):
    results = {}
    for name in present:
        results[name] = {"rc": 0, "value": 1}
    for name in absent:
        results[name] = {"rc": SIGILL, "err": ""}
    for name in error:
        results[name] = {"rc": 1, "err": "PermissionError"}
    return results


def _gcc(accepts=None, emits=True, tunes=("sifive-p400-series",)):
    asm = "f:\n\tsh3add\ta0,a0,a1\n\tret\ng:\n\tandn\ta0,a0,a1\n"
    return {
        "version": "gcc (Debian 14.2.0-19) 14.2.0",
        "accepts": list(
            accepts
            if accepts is not None
            else ("zicond", "zfa", "zfhmin", "zba", "zbb", "zbs", "v")
        ),
        "tunes": list(tunes),
        "flags": ["-mstrict-align", "-flto=auto", "-ffat-lto-objects"],
        "emit_asm": asm if emits else "f:\n\tslli\ta0,a0,3\n",
    }


def _megrez_payload(**overrides):
    payload = {
        "cpuinfo": MEGREZ_CPUINFO,
        "dt_model": "Milk-V Megrez\x00",
        "dt_compatible": "eswin,eic7700\x00riscv\x00",
        "hwprobe": {"pairs": MEGREZ_PAIRS},
        "tests": _tests(
            present=("zba", "zbb"),
            absent=("zbs", "zicond", "zfa", "zfhmin", "v"),
        ),
        "nproc": 4,
        "cpus": [0, 1, 2, 3],
        "meminfo": "MemTotal:       26164504 kB\nMemFree: 1 kB\n",
        "loadavg": "0.12 0.20 0.18 1/180 4242\n",
        "uname": ["Linux", "megrez", "6.6.18-eic7x", "#1 SMP", "riscv64"],
        "perf_event_paranoid": "3\n",
        "perf": None,
        "musl": False,
        "glibc": "glibc 2.41",
        "caches": {"index0": {"level": "1", "type": "Data", "size": "32K"}},
        "cpufreq": {
            "scaling_governor": "performance",
            "scaling_cur_freq": "1800000",
            "scaling_max_freq": "1800000",
            "cpuinfo_max_freq": "1800000",
        },
        "thermal": [52000],
        "compilers": {"gcc": _gcc()},
        "go": "go version go1.26.5 linux/riscv64",
        "rustc": "rustc 1.90.0 (1159e78c4 2025-09-14)",
        "rust_cpus": (
            "Available CPUs for this target:\n"
            "    generic-rv64\n"
            "    sifive-p550\n"
        ),
        # rustc --print cfg, default and with -C target-cpu=sifive-p550.
        "rust_cpu_features": {
            "(default)": ["a", "c", "d", "f", "m"],
            "sifive-p550": ["a", "c", "d", "f", "m", "zba", "zbb"],
        },
    }
    payload.update(overrides)
    return payload


def _megrez():
    return evaluate_probe(_megrez_payload(), "native")


class TestParsing(unittest.TestCase):
    """The raw sources parse into extension sets."""

    def test_isa_string_letters_and_extensions(self) -> None:
        extensions = parse_isa_string(MEGREZ_ISA)
        for name in ("i", "m", "a", "f", "d", "c", "h", "zba", "zbb"):
            self.assertIn(name, extensions)
        self.assertNotIn("zbs", extensions)
        self.assertIn("sscofpmf", extensions)

    def test_g_expands_and_versions_are_dropped(self) -> None:
        """``g`` means imafd + Zicsr + Zifencei; ``2p1`` goes away."""
        extensions = parse_isa_string("rv64i2p1m2p0gc_zba1p0")
        self.assertTrue({"i", "m", "a", "f", "d", "c"} <= extensions)
        self.assertIn("zicsr", extensions)
        self.assertIn("zba", extensions)
        self.assertNotIn("zba1p0", extensions)

    def test_non_riscv_strings_give_nothing(self) -> None:
        """An x86 flag list is not an ISA string."""
        self.assertEqual(parse_isa_string("fpu vme sse"), set())
        self.assertEqual(parse_isa_string(""), set())

    def test_cpuinfo_reads_the_first_cpu_only(self) -> None:
        fields = parse_cpuinfo(MEGREZ_CPUINFO)
        self.assertEqual(fields["isa"], MEGREZ_ISA)
        self.assertEqual(fields["mvendorid"], "0x489")
        self.assertEqual(fields["uarch"], "eswin,eic770x")

    def test_host_x86_cpuinfo_is_rejected(self) -> None:
        """Old QEMU shows the host's cpuinfo; it must not count."""
        self.assertEqual(parse_cpuinfo(X86_CPUINFO), {})
        self.assertEqual(parse_cpuinfo(None), {})

    def test_hwprobe_bits(self) -> None:
        """BASE_BEHAVIOR_IMA and IMA_EXT_0 bits map to extensions."""
        pairs = {3: 1, 4: (1 << 0) | (1 << 2) | (1 << 5) | (1 << 35)}
        self.assertEqual(
            hwprobe_extensions(pairs),
            {"i", "m", "a", "f", "d", "v", "zbs", "zicond"},
        )
        self.assertEqual(hwprobe_extensions({}), set())

    def test_misaligned_key_order(self) -> None:
        """MISALIGNED_SCALAR_PERF wins over CPUPERF_0."""
        self.assertEqual(misaligned_speed({5: 3, 9: 2}), "slow")
        self.assertEqual(misaligned_speed({5: 0x10 | 1}), "emulated")
        self.assertEqual(misaligned_speed({}), "unknown")

    def test_core_match_by_compatible_or_ids(self) -> None:
        """A compatible string, the uarch, or vendor AND arch id match."""
        by_dt = match_core({"compatible": ["sifive,p550", "riscv"]})
        self.assertEqual(by_dt["name"], "sifive-p550")
        by_uarch = match_core({"uarch": "sifive,u74-mc"})
        self.assertEqual(by_uarch["name"], "sifive-u74")
        cases = {
            (0x489, 0x8000000000000008): "sifive-p550",
            (0x489, 0x8000000000000007): "sifive-u74",
            (0x710, 0x8000000058000001): "spacemit-x60",
        }
        for (vendor, arch), name in cases.items():
            with self.subTest(name=name):
                core = {"compatible": [], "mvendorid": vendor,
                        "marchid": arch}
                self.assertEqual(match_core(core)["name"], name)
        # One vendor makes many cores: the vendor id alone never matches.
        self.assertIsNone(match_core({"mvendorid": 0x489}))
        self.assertIsNone(
            match_core({"mvendorid": 0x489, "marchid": 0x8000000000000099})
        )
        # T-Head cores report marchid 0; only `compatible` tells them
        # apart.
        self.assertIsNone(match_core({"mvendorid": 0x5B7, "marchid": 0}))
        self.assertEqual(
            match_core({"compatible": ["thead,c910", "riscv"]})["name"],
            "thead-c910",
        )
        self.assertIsNone(match_core({}))

    def test_tune_candidates_cover_the_table(self) -> None:
        """The probe tests every tune name that the table can choose."""
        for row in machine.CORE_TABLE:
            for name in row["gcc_tunes"] + row["clang_tunes"]:
                self.assertIn(name, machine.TUNE_CANDIDATES)
            if row["rust_cpu"]:
                self.assertIn(row["rust_cpu"], machine.RUST_CPU_CANDIDATES)

    def test_misaligned_timing(self) -> None:
        """The load-time ratio and SIGBUS give the access speed."""
        cases = [
            ({"rc": 0, "aligned_ns": 0.6, "misaligned_ns": 1.5}, "fast"),
            ({"rc": 0, "aligned_ns": 0.6, "misaligned_ns": 6.0}, "slow"),
            ({"rc": 0, "aligned_ns": 0.6, "misaligned_ns": 900.0},
             "emulated"),
            ({"rc": -7}, "unsupported"),
            ({"rc": 135}, "unsupported"),
            ({"rc": 1, "err": "PermissionError"}, "unknown"),
            ({"rc": 0, "aligned_ns": 0, "misaligned_ns": 1.0}, "unknown"),
            ({}, "unknown"),
        ]
        for test, speed in cases:
            with self.subTest(test=test):
                self.assertEqual(misaligned_from_timing(test), speed)

    def test_profile_labels(self) -> None:
        """Full profiles get their name; others list the extras."""
        base = {"i", "m", "a", "f", "d", "c"}
        self.assertEqual(
            profile_label(base | {"zba", "zbb"}), "rv64gc+zba+zbb"
        )
        rva22 = base | set(machine._RVA22_CHECK)
        self.assertEqual(profile_label(rva22), "RVA22U64")
        rva23 = rva22 | set(machine._RVA23_CHECK)
        self.assertEqual(profile_label(rva23), "RVA23U64")
        self.assertEqual(profile_label({"i"}), "below rv64gc")


class TestEvaluateProbe(unittest.TestCase):
    """Raw probe output becomes one machine profile."""

    def test_megrez_profile(self) -> None:
        """The P550 shows Zba and Zbb, no Zbs, no V, slow misalignment."""
        profile = _megrez()
        self.assertEqual(profile.kind, "native")
        self.assertTrue(profile.has("zba", "zbb"))
        self.assertFalse(profile.has("zbs"))
        self.assertFalse(profile.has("v"))
        self.assertIsNone(profile.vlen)
        self.assertEqual(profile.label, "rv64gc+zba+zbb")
        self.assertEqual(profile.core_name, "sifive-p550")
        self.assertEqual(profile.misaligned, "slow")
        self.assertEqual(profile.cores, 4)
        self.assertEqual(profile.allowed_cpus, [0, 1, 2, 3])
        self.assertEqual(profile.mem_kib, 26164504)
        self.assertEqual(profile.kernel, "6.6.18-eic7x")
        self.assertEqual(profile.libc, "glibc 2.41")
        self.assertEqual(profile.board["governor"], "performance")
        self.assertEqual(profile.board["perf_event_paranoid"], 3)
        self.assertEqual(profile.sources, ["cpuinfo", "hwprobe", "tests"])
        self.assertEqual(profile.toolchains["go"], "1.26.5")
        self.assertEqual(profile.toolchains["rust"], "1.90.0")
        self.assertIn("sifive-p550", profile.toolchains["rust_cpus"])
        self.assertTrue(profile.isa_variants_allowed)

    def test_test_instructions_win_over_cpuinfo(self) -> None:
        """A SIGILL removes an extension that cpuinfo claimed."""
        payload = _megrez_payload(
            tests=_tests(present=("zba",), absent=("zbb",))
        )
        profile = evaluate_probe(payload, "native")
        self.assertTrue(profile.has("zba"))
        self.assertFalse(profile.has("zbb"))
        self.assertEqual(profile.tests["zbb"], "absent")

    def test_vector_test_gives_vlen(self) -> None:
        """The V test returns vlenb; VLEN is eight times that."""
        tests = _tests(present=("zba", "zbb"))
        tests["v"] = {"rc": 0, "value": 32}
        profile = evaluate_probe(_megrez_payload(tests=tests), "native")
        self.assertTrue(profile.has("v"))
        self.assertEqual(profile.vlen, 256)

    def test_old_qemu_relies_on_test_instructions(self) -> None:
        """QEMU 6.2: host cpuinfo, no hwprobe; tests find rv64gc only."""
        payload = _megrez_payload(
            cpuinfo=X86_CPUINFO,
            hwprobe={"error": "errno 38"},
            tests=_tests(absent=("zba", "zbb", "zbs", "zicond", "v")),
            dt_compatible=None,
            dt_model=None,
        )
        profile = evaluate_probe(payload, "qemu")
        self.assertEqual(profile.sources, ["tests"])
        self.assertEqual(profile.label, "rv64gc")
        self.assertFalse(profile.isa_variants_allowed)
        text = " ".join(profile.warnings)
        self.assertIn("QEMU older than 8.1", text)
        self.assertIn("riscv_hwprobe is not available", text)
        self.assertIn("--emulate-cpu", text)

    def test_container_seccomp_blocks_hwprobe(self) -> None:
        """Live Megrez result: Linux 6.6 has hwprobe, the container not."""
        payload = _megrez_payload(
            hwprobe={"error": "errno 38"},
            uname=["Linux", "megrez", "6.6.77-win2030", "#1", "riscv64"],
        )
        profile = evaluate_probe(payload, "native")
        self.assertEqual(profile.sources, ["cpuinfo", "tests"])
        self.assertEqual(profile.label, "rv64gc+zba+zbb")
        self.assertEqual(profile.misaligned, "unknown")
        self.assertEqual(profile.core_name, "sifive-p550")
        text = " ".join(profile.warnings)
        self.assertIn("seccomp profile does not list it", text)
        self.assertNotIn("it needs Linux 6.4", text)
        self.assertEqual(align_flags(profile, "gcc"), ["-mstrict-align"])

    def test_timing_test_answers_when_hwprobe_cannot(self) -> None:
        """In the container the load timing gives the misaligned speed."""
        blocked = {
            "hwprobe": {"error": "errno 38"},
            "uname": ["Linux", "megrez", "6.6.77-win2030", "#1", "riscv64"],
        }
        fast = _megrez_payload(
            misaligned_test={"rc": 0, "aligned_ns": 0.56,
                             "misaligned_ns": 1.1},
            **blocked,
        )
        profile = evaluate_probe(fast, "native")
        self.assertEqual(profile.misaligned, "fast")
        self.assertEqual(profile.misaligned_source, "timing test")
        self.assertEqual(align_flags(profile, "gcc"), [])
        self.assertIn("1.10 ns misaligned", "\n".join(
            machine.describe(profile)
        ))
        trap = _megrez_payload(
            misaligned_test={"rc": 0, "aligned_ns": 0.56,
                             "misaligned_ns": 480.0},
            **blocked,
        )
        profile = evaluate_probe(trap, "native")
        self.assertEqual(profile.misaligned, "emulated")
        self.assertEqual(align_flags(profile, "gcc"), ["-mstrict-align"])
        # Under QEMU the timing shows the emulator: it is not used.
        profile = evaluate_probe(fast, "qemu")
        self.assertEqual(profile.misaligned, "unknown")
        self.assertEqual(profile.misaligned_source, "none")

    def test_hwprobe_answer_wins_over_the_timing(self) -> None:
        """The kernel's own measurement comes first."""
        payload = _megrez_payload(
            misaligned_test={"rc": 0, "aligned_ns": 0.5,
                             "misaligned_ns": 0.6},
        )
        profile = evaluate_probe(payload, "native")
        self.assertEqual(profile.misaligned, "slow")
        self.assertEqual(profile.misaligned_source, "hwprobe")

    def test_qemu_max_cpu_turns_isa_variants_off(self) -> None:
        """The default QEMU CPU matches no real board."""
        payload = _megrez_payload(
            cpuinfo=QEMU_MAX_CPUINFO,
            tests=_tests(present=("zba", "zbb", "zbs", "zicond", "v")),
        )
        profile = evaluate_probe(payload, "qemu")
        self.assertTrue(profile.has("zbs", "v"))
        self.assertIsNone(profile.core_name)
        self.assertFalse(profile.isa_variants_allowed)
        self.assertIsNone(march_for(profile, "gcc"))

    def test_named_qemu_cpu_allows_isa_variants(self) -> None:
        """With --emulate-cpu, the emulated set is a real target."""
        payload = _megrez_payload(cpuinfo=QEMU_MAX_CPUINFO)
        profile = evaluate_probe(payload, "qemu", emulated_cpu="rva23u64")
        self.assertTrue(profile.isa_variants_allowed)
        self.assertEqual(profile.emulated_cpu, "rva23u64")

    def test_explicit_max_model_is_no_target(self) -> None:
        """--emulate-cpu max (or max with properties) is QEMU's max."""
        payload = _megrez_payload(cpuinfo=QEMU_MAX_CPUINFO)
        for model in ("max", "MAX,v=true", ""):
            with self.subTest(model=model):
                profile = evaluate_probe(payload, "qemu", emulated_cpu=model)
                self.assertFalse(profile.isa_variants_allowed)
                self.assertIn("max CPU", " ".join(profile.warnings))
        self.assertTrue(machine.is_named_model("rv64i,zba=true"))
        self.assertFalse(machine.is_named_model(None))

    def test_failed_test_instructions_warn(self) -> None:
        """An exec-protected sandbox leaves cpuinfo and hwprobe."""
        payload = _megrez_payload(
            tests=_tests(error=("zba", "zbb", "zbs"))
        )
        profile = evaluate_probe(payload, "native")
        self.assertNotIn("tests", profile.sources)
        self.assertTrue(profile.has("zba", "zbb"))
        self.assertIn("could not run", " ".join(profile.warnings))

    def test_gcc_that_cannot_emit_bitmanip_is_flagged(self) -> None:
        """GCC 11 accepts zba/zbb but emits none of them."""
        payload = _megrez_payload(compilers={"gcc": _gcc(emits=False)})
        profile = evaluate_probe(payload, "native")
        self.assertIn("emits no such instruction", " ".join(
            profile.warnings
        ))
        self.assertIsNone(march_for(profile, "gcc"))

    def test_profile_round_trips_through_a_dict(self) -> None:
        profile = _megrez()
        again = MachineProfile.from_dict(json.loads(json.dumps(
            profile.to_dict()
        )))
        self.assertEqual(again, profile)


class TestRules(unittest.TestCase):
    """The rules turn a profile into flags."""

    def test_march_lists_probed_and_accepted_extensions(self) -> None:
        """The Megrez gets rv64gc_zba_zbb and never Zbs."""
        self.assertEqual(march_for(_megrez(), "gcc"), "rv64gc_zba_zbb")
        self.assertIsNone(march_for(_megrez(), "clang"))

    def test_march_canonical_order_and_vector(self) -> None:
        """Z extensions come in I, F, B order; V joins the letters."""
        tests = _tests(present=("zba", "zbb", "zbs", "zicond", "zfa"))
        tests["v"] = {"rc": 0, "value": 16}
        profile = evaluate_probe(_megrez_payload(tests=tests), "native")
        self.assertEqual(
            march_for(profile, "gcc"),
            "rv64gc_zicond_zfa_zba_zbb_zbs",
        )
        self.assertEqual(
            march_for(profile, "gcc", include_vector=True),
            "rv64gcv_zicond_zfa_zba_zbb_zbs",
        )

    def test_march_needs_compiler_acceptance(self) -> None:
        payload = _megrez_payload(
            compilers={"gcc": _gcc(accepts=("zba",))}
        )
        profile = evaluate_probe(payload, "native")
        self.assertEqual(march_for(profile, "gcc"), "rv64gc_zba")

    def test_tune_prefers_the_core_table_order(self) -> None:
        """sifive-p550 wins when accepted; else the next name."""
        profile = _megrez()
        self.assertEqual(
            tune_for(profile, "gcc"), "-mtune=sifive-p400-series"
        )
        payload = _megrez_payload(compilers={
            "gcc": _gcc(tunes=("sifive-p550", "sifive-p400-series")),
            "clang-19": _gcc(tunes=("sifive-p450",)),
        })
        profile = evaluate_probe(payload, "native")
        self.assertEqual(tune_for(profile, "gcc"), "-mtune=sifive-p550")
        self.assertEqual(
            tune_for(profile, "clang-19"), "-mtune=sifive-p450"
        )

    def test_unknown_core_gets_no_tune(self) -> None:
        payload = _megrez_payload(
            cpuinfo=MEGREZ_CPUINFO.replace("0x489", "0x5b7"),
            hwprobe={"pairs": [[3, 1], [4, 0b11011], [0, 0x5B7]]},
        )
        profile = evaluate_probe(payload, "native")
        self.assertIsNone(profile.core_name)
        self.assertIsNone(tune_for(profile, "gcc"))

    def test_strict_align_unless_fast(self) -> None:
        profile = _megrez()
        self.assertEqual(align_flags(profile, "gcc"), ["-mstrict-align"])
        profile.misaligned = "fast"
        self.assertEqual(align_flags(profile, "gcc"), [])
        profile.misaligned = "unknown"
        self.assertEqual(align_flags(profile, "missing"), [])

    def test_go_levels(self) -> None:
        """Strict levels, plus the no-Zbs exception up to Go 1.26."""
        profile = _megrez()
        level, note = goriscv64_for(profile)
        self.assertEqual(level, "rva22u64")
        self.assertIn("lacks Zbs", note)
        profile.toolchains["go"] = "1.27.0"
        self.assertEqual(goriscv64_for(profile), ("rva20u64", None))
        profile.toolchains["go"] = "1.22.4"
        self.assertEqual(goriscv64_for(profile), ("rva20u64", None))
        profile.toolchains["go"] = "1.26.5"
        profile.extensions = sorted(set(profile.extensions) | {"zbs"})
        self.assertEqual(goriscv64_for(profile), ("rva22u64", None))
        profile.extensions = sorted(
            set(profile.extensions) | {"v", "zicond", "zfa"}
        )
        self.assertEqual(goriscv64_for(profile), ("rva23u64", None))

    def test_rust_flags(self) -> None:
        """A known CPU that fits wins; else the stable features."""
        profile = _megrez()
        self.assertEqual(
            rust_flags_for(profile), "-C target-cpu=sifive-p550"
        )
        profile.toolchains["rust_cpus"] = ["generic-rv64"]
        self.assertEqual(
            rust_flags_for(profile), "-C target-feature=+zba,+zbb"
        )
        profile.toolchains["rust"] = None
        self.assertIsNone(rust_flags_for(profile))

    def test_rust_cpu_must_fit_the_machine(self) -> None:
        features = {
            "(default)": ["a", "c", "d", "f", "m"],
            "sifive-p550": ["a", "c", "d", "f", "m", "zba", "zbb", "zbs"],
        }
        profile = _megrez()
        profile.toolchains["rust_cpu_features"] = features
        self.assertFalse(machine.rust_cpu_fits(profile, "sifive-p550"))
        self.assertEqual(
            rust_flags_for(profile), "-C target-feature=+zba,+zbb"
        )
        features["sifive-p550"] = ["a", "c", "d", "f", "m", "zba",
                                   "unaligned-scalar-mem"]
        self.assertFalse(machine.rust_cpu_fits(profile, "sifive-p550"))
        profile.misaligned = "fast"
        self.assertTrue(machine.rust_cpu_fits(profile, "sifive-p550"))
        # Without rustc's feature lists, no CPU model is used.
        profile.toolchains["rust_cpu_features"] = {}
        self.assertFalse(machine.rust_cpu_fits(profile, "sifive-p550"))

    def test_forbidden_classes_follow_the_profile(self) -> None:
        """The Megrez cannot run Zbs, Zicond, Zfa, Zfh, V or Zcb."""
        self.assertEqual(
            forbidden_classes(_megrez()),
            ["zbs", "zicond", "zfa", "zfh", "v", "zcb"],
        )

    def test_qemu_model_of_the_megrez(self) -> None:
        """A bare rv64i CPU plus the probed extensions."""
        model = qemu_cpu_for(_megrez())
        self.assertTrue(model.startswith("rv64i,m=true,a=true"))
        self.assertIn("zba=true", model)
        self.assertIn("zbb=true", model)
        self.assertNotIn("zbs=true", model)
        self.assertNotIn("v=true", model)

    def test_rules_summary_for_probe_output(self) -> None:
        """--probe lists the flags, Go level, Rust flags and QEMU model."""
        lines = describe_rules(_megrez())
        text = "\n".join(lines)
        # The fixture's GCC 14 knows sifive-p400-series, not sifive-p550.
        self.assertIn(
            "gcc: -mtune=sifive-p400-series -mstrict-align "
            "-march=rv64gc_zba_zbb",
            text,
        )
        self.assertIn("GORISCV64=rva22u64 (the machine lacks Zbs", text)
        self.assertIn("RUSTFLAGS=-C target-cpu=sifive-p550", text)
        self.assertTrue(lines[-1].startswith("QEMU model:  --emulate-cpu"))
        qemu = evaluate_probe(_megrez_payload(), "qemu")
        self.assertNotIn("QEMU model", "\n".join(describe_rules(qemu)))


class TestIntegrityCheck(unittest.TestCase):
    """The machine check: unchanged code must give the same data."""

    def test_program_runs_on_this_host(self) -> None:
        """Every CPU child reports its wrong round trips (none here)."""
        import os
        import subprocess
        import sys

        ast.parse(machine._INTEGRITY_PROGRAM)
        cpus = sorted(os.sched_getaffinity(0))[:2]
        done = subprocess.run(
            [sys.executable, "-c", machine._INTEGRITY_PROGRAM,
             json.dumps({"cpus": cpus + [99999], "rounds": 2,
                         "size_mb": 1})],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        payload = machine._parse_probe_output(done.stdout)
        # A CPU outside the affinity set is dropped.
        self.assertEqual(payload["bad"], {str(cpu): 0 for cpu in cpus})
        self.assertEqual((payload["rounds"], payload["size_mb"]), (2, 1))
        self.assertEqual(machine.integrity_failures(payload), {})

    def test_failures_and_crashes_count(self) -> None:
        """Wrong data and a crashed child both fail the check."""
        result = {"bad": {"0": 0, "1": 2, "2": -1, "3": 0}}
        self.assertEqual(machine.integrity_failures(result),
                         {"1": 2, "2": -1})

    def test_check_reports_a_broken_run(self) -> None:
        """No JSON line: the check gives the error, not a verdict."""
        broken = CommandResult("python3", 1, "", "Traceback", 0.1)
        with mock.patch.object(machine, "execute_command",
                               return_value=broken):
            result, error = machine.check_integrity([0, 1])
        self.assertIsNone(result)
        self.assertIn("Traceback", error)


class TestProbeMachine(unittest.TestCase):
    """probe_machine runs one program in the sandbox."""

    def test_probe_program_parses(self) -> None:
        """The program and its child code are valid Python."""
        ast.parse(machine._PROBE_PROGRAM)
        tree = ast.parse(machine._PROBE_PROGRAM)
        child = next(
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "CHILD"
        )
        ast.parse(child)

    def test_probe_runs_unvalidated_with_the_qemu_cpu(self) -> None:
        """QEMU_CPU reaches the sandbox only for QEMU runs."""
        output = machine._OUTPUT_MARKER + json.dumps(_megrez_payload())
        ok = CommandResult("python3", 0, output, "", 0.5)
        with mock.patch.object(
            machine, "execute_command", return_value=ok
        ) as execute:
            profile, error = probe_machine("qemu", "rv64i,zba=true")
        self.assertIsNone(error)
        self.assertEqual(profile.emulated_cpu, "rv64i,zba=true")
        kwargs = execute.call_args.kwargs
        self.assertFalse(kwargs["validate"])
        self.assertEqual(kwargs["extra_env"], {"QEMU_CPU": "rv64i,zba=true"})
        self.assertEqual(kwargs["cwd"], "/workspace")
        argv = execute.call_args.args[0]
        self.assertEqual(argv[:2], ["python3", "-c"])
        self.assertIn("sifive-p550", json.loads(argv[3])["tune_candidates"])

    def test_native_probe_ignores_the_qemu_cpu(self) -> None:
        output = machine._OUTPUT_MARKER + json.dumps(_megrez_payload())
        ok = CommandResult("python3", 0, output, "", 0.5)
        with mock.patch.object(
            machine, "execute_command", return_value=ok
        ) as execute:
            probe_machine("native", "rva23u64")
        self.assertIsNone(execute.call_args.kwargs["extra_env"])

    def test_missing_output_is_an_error(self) -> None:
        """A probe that cannot start returns its exit code."""
        failed = CommandResult("python3", 127, "", "not found", 0.1)
        with mock.patch.object(
            machine, "execute_command", return_value=failed
        ):
            profile, error = probe_machine("native")
        self.assertIsNone(profile)
        self.assertIn("exit 127", error)


if __name__ == "__main__":
    unittest.main()
