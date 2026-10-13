"""Tests for src/optimize.py: the tunes, replay, gates and flow.

The last classes run the in-sandbox programs for real on this host and
feed a real result to the report and the zip.
"""

import contextlib
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src import bench, optimize
from src import workload as workload_module
from src.machine import MachineProfile
from src.optimize import (
    TREE_TOKEN,
    Variant,
    build_flags,
    build_ladder,
    c_ladder,
    classify_segment,
    gate_build_facts,
    gate_tests,
    go_ladder,
    language_of,
    redundant_tunes,
    replay_commands,
    replayable,
    rewrite_for_tree,
    run_optimization,
    rust_ladder,
    variant_env,
)
from src.state import (
    BuildPhase,
    BuildPlan,
    BuildStatus,
    CommandResult,
    create_initial_state,
)

REPO = "/workspace/repos/zlib"
ROOT = REPO + ".opt"

# (class, instruction, encoding as little-endian hex). The encodings come
# from llvm-mc 19 (-mattr with every extension and -riscv-no-aliases).
# None marks base-ISA instructions, which the scan must not count.
ENCODINGS = (
    ("zba", "sh1add a0, a1, a2", "33a5c520"),
    ("zba", "sh2add a0, a1, a2", "33c5c520"),
    ("zba", "sh3add a0, a1, a2", "33e5c520"),
    ("zba", "add.uw a0, a1, a2", "3b85c508"),
    ("zba", "sh1add.uw a0, a1, a2", "3ba5c520"),
    ("zba", "sh2add.uw a0, a1, a2", "3bc5c520"),
    ("zba", "sh3add.uw a0, a1, a2", "3be5c520"),
    ("zba", "slli.uw a0, a1, 3", "1b953508"),
    ("zbb", "andn a0, a1, a2", "33f5c540"),
    ("zbb", "orn a0, a1, a2", "33e5c540"),
    ("zbb", "xnor a0, a1, a2", "33c5c540"),
    ("zbb", "clz a0, a1", "13950560"),
    ("zbb", "ctz a0, a1", "13951560"),
    ("zbb", "cpop a0, a1", "13952560"),
    ("zbb", "clzw a0, a1", "1b950560"),
    ("zbb", "ctzw a0, a1", "1b951560"),
    ("zbb", "cpopw a0, a1", "1b952560"),
    ("zbb", "max a0, a1, a2", "33e5c50a"),
    ("zbb", "maxu a0, a1, a2", "33f5c50a"),
    ("zbb", "min a0, a1, a2", "33c5c50a"),
    ("zbb", "minu a0, a1, a2", "33d5c50a"),
    ("zbb", "sext.b a0, a1", "13954560"),
    ("zbb", "sext.h a0, a1", "13955560"),
    ("zbb", "zext.h a0, a1", "3bc50508"),
    ("zbb", "rol a0, a1, a2", "3395c560"),
    ("zbb", "ror a0, a1, a2", "33d5c560"),
    ("zbb", "rori a0, a1, 7", "13d57560"),
    ("zbb", "rolw a0, a1, a2", "3b95c560"),
    ("zbb", "rorw a0, a1, a2", "3bd5c560"),
    ("zbb", "roriw a0, a1, 7", "1bd57560"),
    ("zbb", "orc.b a0, a1", "13d57528"),
    ("zbb", "rev8 a0, a1", "13d5856b"),
    ("zbs", "bclr a0, a1, a2", "3395c548"),
    ("zbs", "bclri a0, a1, 40", "1395854a"),
    ("zbs", "bext a0, a1, a2", "33d5c548"),
    ("zbs", "bexti a0, a1, 40", "13d5854a"),
    ("zbs", "binv a0, a1, a2", "3395c568"),
    ("zbs", "binvi a0, a1, 40", "1395856a"),
    ("zbs", "bset a0, a1, a2", "3395c528"),
    ("zbs", "bseti a0, a1, 40", "1395852a"),
    ("zbc", "clmul a0, a1, a2", "3395c50a"),
    ("zbc", "clmulh a0, a1, a2", "33b5c50a"),
    ("zbc", "clmulr a0, a1, a2", "33a5c50a"),
    ("zicond", "czero.eqz a0, a1, a2", "33d5c50e"),
    ("zicond", "czero.nez a0, a1, a2", "33f5c50e"),
    ("zfa", "fli.d fa0, 1.0", "530518f2"),
    ("zfa", "fli.s fa0, 0.5", "530516f0"),
    ("zfa", "fminm.d fa0, fa1, fa2", "53a5c52a"),
    ("zfa", "fmaxm.s fa0, fa1, fa2", "53b5c528"),
    ("zfa", "fround.d fa0, fa1", "53f54542"),
    ("zfa", "froundnx.s fa0, fa1", "53f55540"),
    ("zfa", "fcvtmod.w.d a0, fa1, rtz", "539585c2"),
    ("zfa", "fleq.d a0, fa1, fa2", "53c5c5a2"),
    ("zfa", "fltq.s a0, fa1, fa2", "53d5c5a0"),
    ("zfh", "fadd.h fa0, fa1, fa2", "53f5c504"),
    ("zfh", "fmul.h fa0, fa1, fa2", "53f5c514"),
    ("zfh", "flh fa0, 8(a0)", "07158500"),
    ("zfh", "fsh fa0, 8(a0)", "2714a500"),
    ("zfh", "fcvt.s.h fa0, fa1", "53852540"),
    ("zfh", "fcvt.d.h fa0, fa1", "53852542"),
    ("zfh", "fcvt.h.s fa0, fa1", "53f50544"),
    ("zfh", "fmv.x.h a0, fa1", "538505e4"),
    ("zfh", "fcvt.h.w fa0, a1", "53f505d4"),
    ("v", "vsetvli t0, a0, e32, m1, ta, ma", "d772050d"),
    ("v", "vadd.vv v1, v2, v3", "d7802102"),
    ("v", "vle32.v v1, (a0)", "87600502"),
    ("v", "vse32.v v1, (a0)", "a7600502"),
    ("v", "vle8.v v1, (a0)", "87000502"),
    ("v", "vle64.v v1, (a0)", "87700502"),
    ("v", "vle16.v v1, (a0)", "87500502"),
    ("zcb", "c.lbu a0, 1(a1)", "c881"),
    ("zcb", "c.lhu a0, 2(a1)", "a885"),
    ("zcb", "c.lh a0, 2(a1)", "e885"),
    ("zcb", "c.sb a0, 1(a1)", "c889"),
    ("zcb", "c.sh a0, 2(a1)", "a88d"),
    ("zcb", "c.zext.b a0", "619d"),
    ("zcb", "c.sext.b a0", "659d"),
    ("zcb", "c.zext.h a0", "699d"),
    ("zcb", "c.sext.h a0", "6d9d"),
    ("zcb", "c.zext.w a0", "719d"),
    ("zcb", "c.not a0", "759d"),
    ("zcb", "c.mul a0, a1", "4d9d"),
    (None, "add a0, a1, a2", "3385c500"),
    (None, "sub a0, a1, a2", "3385c540"),
    (None, "sra a0, a1, a2", "33d5c540"),
    (None, "srai a0, a1, 40", "13d58542"),
    (None, "slli a0, a1, 40", "13958502"),
    (None, "srli a0, a1, 40", "13d58502"),
    (None, "mul a0, a1, a2", "3385c502"),
    (None, "div a0, a1, a2", "33c5c502"),
    (None, "addw a0, a1, a2", "3b85c500"),
    (None, "subw a0, a1, a2", "3b85c540"),
    (None, "sraiw a0, a1, 7", "1bd57540"),
    (None, "slliw a0, a1, 7", "1b957500"),
    (None, "c.lw a0, 8(a1)", "8845"),
    (None, "c.sd a0, 8(a1)", "88e5"),
    (None, "fadd.d fa0, fa1, fa2", "53f5c502"),
    (None, "fadd.s fa0, fa1, fa2", "53f5c500"),
    (None, "flw fa0, 8(a0)", "07258500"),
    (None, "c.fld fa0, 8(a0)", "0825"),
    (None, "c.fsd fa0, 8(a0)", "08a5"),
    (None, "fcvt.s.d fa0, fa1", "53f51540"),
    (None, "fcvt.d.s fa0, fa1", "53850542"),
    (None, "feq.d a0, fa1, fa2", "53a5c5a2"),
    (None, "flt.d a0, fa1, fa2", "5395c5a2"),
    (None, "fle.d a0, fa1, fa2", "5385c5a2"),
    (None, "fmv.x.d a0, fa1", "538505e2"),
    (None, "fsgnj.d fa0, fa1, fa2", "5385c522"),
    (None, "fmin.d fa0, fa1, fa2", "5385c52a"),
    (None, "fmax.s fa0, fa1, fa2", "5395c528"),
    (None, "fmadd.d fa0, fa1, fa2, fa3", "43f5c56a"),
    (None, "fcvt.w.d a0, fa1, rtz", "539505c2"),
    (None, "amoadd.w a0, a1, (a2)", "2f25b600"),
    (None, "lr.d a0, (a1)", "2fb50510"),
    (None, "sc.d a0, a1, (a2)", "2f35b618"),
    (None, "csrr a0, vlenb", "732520c2"),
    (None, "ecall", "73000000"),
    (None, "fence", "0f00f00f"),
    (None, "c.addi a0, 1", "0505"),
    (None, "c.lw a0, 4(a1)", "c841"),
    (None, "c.ld a0, 8(a1)", "8865"),
    (None, "c.subw a0, a1", "0d9d"),
    (None, "c.addw a0, a1", "2d9d"),
    (None, "c.and a0, a1", "6d8d"),
    (None, "c.sub a0, a1", "0d8d"),
    (None, "c.srai a0, 3", "0d85"),
    (None, "c.j 8", "21a0"),
    (None, "c.nop", "0100"),
)


def _gcc(family="gcc", tunes=("sifive-p550", "sifive-p400-series")):
    return {
        "family": family,
        "version": f"{family} test",
        "accepts": ["zicond", "zfa", "zfhmin", "zba", "zbb", "zbs", "v"],
        "tunes": list(tunes),
        "flags": ["-mstrict-align", "-flto=auto", "-ffat-lto-objects"],
        "emits_bitmanip": True,
    }


def _profile(**overrides):
    """Return a Megrez-like profile: Zba and Zbb, no Zbs, no V."""
    data = dict(
        kind="native",
        emulated_cpu=None,
        extensions=sorted(
            {"i", "m", "a", "f", "d", "c", "zicsr", "zifencei", "zba", "zbb"}
        ),
        core_name="sifive-p550",
        misaligned="slow",
        cores=4,
        allowed_cpus=[0, 1, 2, 3],
        toolchains={
            "compilers": {
                "gcc": _gcc(),
                "clang-19": _gcc("clang", ("sifive-p550", "sifive-p450")),
            },
            "go": "1.26.5",
            "rust": "1.90.0",
            "rust_cpus": ["generic-rv64", "sifive-p550"],
            "rust_cpu_features": {
                "(default)": ["a", "c", "d", "f", "m"],
                "sifive-p550": ["a", "c", "d", "f", "m", "zba", "zbb"],
            },
        },
        label="rv64gc+zba+zbb",
    )
    data.update(overrides)
    return MachineProfile(**data)


def _state(build_system="cmake", verified=True, tests="passed"):
    state = create_initial_state("https://github.com/madler/zlib")
    commands = [
        ["apt-get install -y cmake"],
        ["cmake -S . -B build", "cmake --build build -j4"],
    ]
    state.build_plan = BuildPlan(
        build_system=build_system,
        build_system_confidence=0.9,
        phases=[
            BuildPhase(index + 1, f"phase{index}", list(group))
            for index, group in enumerate(commands)
        ],
        total_estimated_duration="1m",
    )
    for group in commands:
        for command in group:
            state.cache_command_result(
                command, CommandResult(command, 0, "", "", 1.0)
            )
    state.build_status = BuildStatus.SUCCESS
    state.artifact_verification = {
        "status": "verified" if verified else "unverified"
    }
    state.curated_artifacts = [
        {
            "filepath": f"{REPO}/build/minigzip",
            "type": "binary",
            "role": "primary",
        },
        {
            "filepath": f"{REPO}/build/libz.so.1",
            "type": "library_shared",
            "role": "primary",
        },
        {
            "filepath": f"{REPO}/build/libz.a",
            "type": "library_static",
            "role": "primary",
        },
        {"filepath": "/usr/local/bin/outside", "type": "binary"},
    ]
    state.package_tests = {
        "status": tests,
        "framework": "ctest",
        "command": "ctest --test-dir build",
    }
    return state


def _uleb(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _elf(code: bytes, arch="rv64i2p1_m2p0_zba1p0", unaligned=1) -> bytes:
    """Return a minimal riscv64 ELF with one code section and attributes."""
    attrs = _uleb(5) + arch.encode() + b"\0" + _uleb(6) + _uleb(unaligned)
    sub = _uleb(1) + struct.pack("<I", 1 + 4 + len(attrs)) + attrs
    vendor = b"riscv\0"
    section = struct.pack("<I", 4 + len(vendor) + len(sub)) + vendor + sub
    attributes = b"A" + section
    text_offset = 64
    attr_offset = text_offset + len(code)
    shoff = attr_offset + len(attributes)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        1,
        243,
        1,
        0,
        0,
        shoff,
        0,
        64,
        0,
        0,
        64,
        3,
        0,
    )
    null = b"\0" * 64
    text = struct.pack(
        "<IIQQQQIIQQ", 0, 1, 6, 0, text_offset, len(code), 0, 0, 2, 0
    )
    attr = struct.pack(
        "<IIQQQQIIQQ",
        0,
        0x70000003,
        0,
        0,
        attr_offset,
        len(attributes),
        0,
        0,
        1,
        0,
    )
    return header + code + attributes + null + text + attr


def _code(*instructions: str) -> bytes:
    table = {name: bytes.fromhex(word) for _, name, word in ENCODINGS}
    return b"".join(table[name] for name in instructions)


def _ar(members) -> bytes:
    """Return an ar archive of (name, body) members."""
    out = b"!<arch>\n"
    for name, body in members:
        header = f"{name:<16}{0:<12}{0:<6}{0:<6}{644:<8}{len(body):<10}`\n"
        out += header.encode() + body + (b"\n" if len(body) % 2 else b"")
    return out


def _inspect_namespace():
    """Load the inspect program without running its trailing main()."""
    namespace = {}
    source = optimize._INSPECT_PROGRAM.replace("\nmain()\n", "\n")
    with mock.patch.object(sys, "argv", ["inspect", "{}"]):
        exec(compile(source, "inspect", "exec"), namespace)
    return namespace


class TestLadder(unittest.TestCase):
    """The probe decides which tunes exist and what they set.

    The class also covers the build-flag scan, the redundant tunes and
    the result text of a tune.
    """

    def test_megrez_c_ladder_is_cumulative(self) -> None:
        """Core, ISA, -O3 and LTO stack up; clang gets its own flags."""
        ladder = c_ladder(_profile())
        self.assertEqual(
            [(v.vid, v.label, v.technique) for v in ladder],
            [
                ("tune1", "Tune-1", "core tuning"),
                ("tune2", "Tune-2", "machine ISA"),
                ("tune3", "Tune-3", "optimization level"),
                ("tune4", "Tune-4", "link-time optimization"),
                ("tune5", "Tune-5", "compiler"),
            ],
        )
        by_id = {v.vid: v for v in ladder}
        self.assertEqual(
            by_id["tune1"].flags, ["-mtune=sifive-p550", "-mstrict-align"]
        )
        self.assertEqual(
            by_id["tune1"].value, "-mtune=sifive-p550 -mstrict-align"
        )
        self.assertEqual(by_id["tune2"].flags[-1], "-march=rv64gc_zba_zbb")
        self.assertEqual(by_id["tune2"].value, "-march=rv64gc_zba_zbb")
        self.assertEqual(by_id["tune3"].flags[-1], "-O3")
        self.assertIn("-ffat-lto-objects", by_id["tune4"].flags)
        self.assertEqual(by_id["tune5"].compiler, "clang-19")
        self.assertEqual(by_id["tune5"].value, "clang-19")
        self.assertEqual(
            by_id["tune5"].flags,
            [
                "-mtune=sifive-p550",
                "-mstrict-align",
                "-march=rv64gc_zba_zbb",
                "-O3",
            ],
        )
        self.assertEqual(
            [v.keeps for v in ladder],
            [None, "Tune-1", "Tune-2", "Tune-3", "Tune-3"],
        )

    def test_vector_machine_gets_a_vector_tune(self) -> None:
        """With V, a vector ISA tune follows the scalar ISA tune."""
        profile = _profile(
            extensions=sorted(set(_profile().extensions) | {"v"}), vlen=256
        )
        ladder = c_ladder(profile)
        techniques = [v.technique for v in ladder]
        self.assertEqual(
            techniques[:3], ["core tuning", "machine ISA", "vector ISA"]
        )
        vector = ladder[2]
        self.assertEqual(vector.value, "-march=rv64gcv_zba_zbb")
        self.assertEqual(vector.note, "VLEN 256")
        o3 = ladder[3]
        self.assertIn("-march=rv64gcv_zba_zbb", o3.flags)
        self.assertNotIn("-march=rv64gc_zba_zbb", o3.flags)
        self.assertIn("-march=rv64gcv_zba_zbb", ladder[-1].flags)
        self.assertLessEqual(
            len(build_ladder(profile, "c")), optimize.MAX_VARIANTS - 1
        )

    def test_qemu_max_and_unknown_core_drop_steps(self) -> None:
        """No tune for an unknown core; no ISA step under QEMU's max."""
        profile = _profile(kind="qemu", core_name=None)
        ladder = c_ladder(profile)
        self.assertEqual(
            [v.technique for v in ladder],
            ["optimization level", "link-time optimization", "compiler"],
        )
        # The first tune shows the alignment flag it brings.
        self.assertEqual(ladder[0].value, "-O3 -mstrict-align")
        self.assertEqual(ladder[0].label, "Tune-1")
        fast = _profile(misaligned="fast")
        self.assertNotIn("-mstrict-align", c_ladder(fast)[0].flags)
        no_gcc = _profile(toolchains={"compilers": {}})
        self.assertEqual(c_ladder(no_gcc), [])

    def test_go_and_rust_ladders(self) -> None:
        """Go gets rva22u64 with the Zbs note; Rust gets the P550 CPU."""
        go = go_ladder(_profile())
        self.assertEqual(go[0].env, {"GORISCV64": "rva22u64"})
        self.assertEqual(
            (go[0].label, go[0].technique), ("Tune-1", "machine ISA")
        )
        self.assertIn("Zbs", go[0].note)
        self.assertEqual(go_ladder(_profile(toolchains={})), [])
        rust = rust_ladder(_profile())
        self.assertEqual(
            rust[0].env, {"RUSTFLAGS": "-C target-cpu=sifive-p550"}
        )
        self.assertEqual(rust[0].technique, "target CPU")
        self.assertEqual(rust[1].env["CARGO_PROFILE_RELEASE_LTO"], "fat")
        self.assertEqual(rust[1].env["RUSTFLAGS"], "-C target-cpu=sifive-p550")
        self.assertEqual((rust[1].label, rust[1].keeps), ("Tune-2", "Tune-1"))
        self.assertEqual(language_of("cargo"), "rust")
        self.assertEqual(language_of("autotools"), "c")
        self.assertEqual(len(build_ladder(_profile(), "c")), 5)

    def test_unconfirmed_names_its_reason(self) -> None:
        """Only a second timing that disagrees "did not hold"."""
        budget = {
            "variant": {"label": "Tune-6"},
            "verdict": "not confirmed",
            "ratio": 0.5343,
            "unchecked": "time",
            "reason": "the time budget ended before the second check",
        }
        held = {
            "variant": {"label": "Tune-3"},
            "verdict": "not confirmed",
            "ratio": 0.9,
            "second_check": {"verdict": "no change"},
            "reason": "the gain did not hold in a second code layout",
        }
        no_base = {
            "verdict": "not confirmed",
            "ratio": 0.9,
            "second_check": {"verdict": None},
            "reason": "no second-layout baseline: build failed",
        }
        self.assertEqual(
            optimize.tune_result(budget),
            "46.6% faster (not confirmed: the time budget ended before "
            "the second check)",
        )
        self.assertEqual(
            optimize.tune_result(held),
            "10.0% faster (did not hold in the second check)",
        )
        self.assertEqual(
            optimize.tune_result(no_base),
            "10.0% faster (not confirmed: no second-layout baseline: build "
            "failed)",
        )
        self.assertEqual(
            optimize.headline(
                {"status": "not confirmed", "variants": [held, budget]}
            ),
            "NO: no faster tune was confirmed (best: Tune-6, 46.6% faster "
            "(not confirmed: the time budget ended before the second "
            "check))",
        )

    def test_tune_result_and_headline(self) -> None:
        """The summary gives a percent, or why there is none."""
        cases = [
            ({"verdict": "faster", "ratio": 0.892}, "10.8% faster"),
            ({"verdict": "slower", "ratio": 1.05}, "5.0% slower"),
            (
                {"verdict": "no change", "ratio": 0.977},
                "2.3% faster (inside the noise)",
            ),
            (
                {"verdict": "not confirmed", "ratio": 0.9},
                "10.0% faster (did not hold in the second check)",
            ),
            (
                {
                    "verdict": "same as tune2",
                    "same_as_label": "Tune-2",
                    "ratio": 0.97,
                },
                "same code as Tune-2, 3.0% faster",
            ),
            ({"verdict": "measurement failed"}, "not timed"),
            ({"verdict": "skipped"}, "skipped"),
            (
                {"verdict": "no effect"},
                "no effect (same code as the baseline)",
            ),
            ({"verdict": "checked"}, "built and checked (QEMU: no time)"),
        ]
        for record, text in cases:
            with self.subTest(verdict=record["verdict"]):
                self.assertEqual(optimize.tune_result(record), text)
        accepted = {
            "status": "accepted",
            "accepted_label": "Tune-6",
            "accepted_title": "clang-19 -O3 + PGO",
            "accepted_ratio": 0.892,
        }
        self.assertEqual(
            optimize.headline(accepted),
            "YES: Tune-6 (clang-19 -O3 + PGO) is 10.8% faster than the "
            "verified build",
        )
        no_gain = {
            "status": "no gain",
            "variants": [
                {
                    "variant": {"label": "Tune-1"},
                    "verdict": "no change",
                    "ratio": 1.017,
                },
                {
                    "variant": {"label": "Tune-2"},
                    "verdict": "no change",
                    "ratio": 0.988,
                },
            ],
        }
        self.assertEqual(
            optimize.headline(no_gain),
            "NO: no tune was clearly faster than the verified build "
            "(best: Tune-2, 1.2% faster (inside the noise))",
        )
        self.assertTrue(
            optimize.headline({"status": "checked"}).startswith("NOT TIMED:")
        )
        self.assertEqual(
            optimize.headline({"status": "error"}),
            "NO: performance mode stopped with an error",
        )

    def test_build_flags_from_the_wrapper_log(self) -> None:
        """Source compiles count; configure checks do not."""
        lines = [
            "-O3 -DNDEBUG -fPIC -c /w/zlib.opt/base/adler32.c -o a.o",
            "-O3 -c ../deflate.c -o d.o",
            "-c CMakeFiles/CMakeScratch/TryCompile-x/src.c -o t.o",
            "conftest.c -o conftest",
            "-shared -o libz.so a.o d.o",
        ]
        flags = build_flags(lines)
        self.assertEqual(
            flags, {"compiles": 2, "levels": ["-O3"], "lto": False}
        )
        mixed = build_flags(
            lines + ["-Os -flto=auto -c small.c -o s.o", "-c plain.c -o p.o"]
        )
        self.assertEqual(mixed["levels"], ["-O0", "-O3", "-Os"])
        self.assertTrue(mixed["lto"])
        self.assertEqual(build_flags(['-c "odd.c'])["compiles"], 1)
        # CMake's own compiler checks, also outside the tree, do not
        # count; the tree's sources do.
        cmake = [
            "-mcmodel=medany -v -o CMakeFiles/cmTC_560d6.dir/"
            "CMakeCCompilerABI.c.o -c /usr/share/cmake-3.31/Modules/"
            "CMakeCCompilerABI.c",
            "-O3 -c /usr/share/other/probe.c -o p.o",
            "-O3 -DNDEBUG -c /w/zlib.opt/base/adler32.c -o a.o",
        ]
        self.assertEqual(
            build_flags(cmake, "/w/zlib.opt/base"),
            {"compiles": 1, "levels": ["-O3"], "lto": False},
        )
        self.assertEqual(build_flags(cmake)["compiles"], 2)

    def test_redundant_tunes(self) -> None:
        """-O3 everywhere skips the -O3 tune; -flto skips LTO."""
        ladder = c_ladder(_profile())
        reasons = redundant_tunes(
            ladder, {"compiles": 15, "levels": ["-O3"], "lto": True}
        )
        self.assertEqual(sorted(reasons), ["tune3", "tune4"])
        self.assertIn("all 15 sources with -O3", reasons["tune3"])
        self.assertEqual(
            redundant_tunes(
                ladder,
                {"compiles": 15, "levels": ["-O2", "-O3"], "lto": False},
            ),
            {},
        )
        self.assertEqual(
            redundant_tunes(
                ladder, {"compiles": 0, "levels": [], "lto": False}
            ),
            {},
        )

    def test_result_text(self) -> None:
        """Each verdict gets one plain line."""
        measured = {
            "verdict": "faster",
            "ratio": 0.892,
            "measurement": {"comparison": {"ci_low": 0.89, "ci_high": 0.894}},
        }
        self.assertEqual(
            optimize.result_text(measured),
            "10.8% faster: time ratio 0.892 (95% CI 0.89-0.894)",
        )
        slower = dict(measured, verdict="slower", ratio=1.05)
        self.assertTrue(
            optimize.result_text(slower).startswith(
                "5.0% slower: time ratio 1.05"
            )
        )
        same = {
            "verdict": "no change",
            "ratio": 1.017,
            "measurement": {"comparison": {"ci_low": 1.007, "ci_high": 1.021}},
        }
        self.assertEqual(
            optimize.result_text(same),
            "1.7% slower, inside the noise: time ratio 1.017 (95% CI "
            "1.007-1.021)",
        )
        twin = {
            "verdict": "same as tune2",
            "same_as_label": "Tune-2",
            "reason": "result reused (no change)",
        }
        self.assertEqual(
            optimize.result_text(twin),
            "same code as Tune-2: result reused (no change)",
        )
        failed = {
            "verdict": "build failed",
            "reason": "`make` exited 2:\nerror one\nerror two",
        }
        self.assertEqual(
            optimize.result_text(failed),
            "build failed: `make` exited 2: error one error two",
        )
        self.assertEqual(
            optimize.result_text({"verdict": "gate 2 failed", "reason": "x"}),
            "package tests failed: x",
        )


class TestReplay(unittest.TestCase):
    """Only the build steps of the verified build run again."""

    def test_segment_rules(self) -> None:
        """Installs, source steps and writes outside the tree drop."""
        cases = {
            "apt-get install -y cmake": "drop",
            "DEBIAN_FRONTEND=noninteractive apt-get update": "drop",
            "sudo make install": "drop",
            "make -j4 install": "drop",
            "make installcheck": "keep",
            "cmake --install build": "drop",
            "go install ./cmd/x": "drop",
            "git fetch origin c_master:c_master": "drop",
            "git submodule update --init": "drop",
            "wget -q https://go.dev/dl/go.tar.gz": "drop",
            "tar -C /usr/local -xzf go.tar.gz": "drop",
            "cp build/tool /usr/local/bin/": "drop",
            "cp build/tool /usr/local/bin 2>/dev/null || true": "drop",
            "cp /usr/share/automake-1.16/config.guess ./": "keep",
            "go build -o /usr/local/bin/x .": "drop",
            "go build -o ./bin/x .": "keep",
            f"go build -o {REPO}/bin/x .": "keep",
            "cmake -S . -B /tmp/out": "drop",
            "cmake --build build -j4": "keep",
            "cd /tmp": "leave",
            "cd ~/src": "leave",
            f"cd {REPO}/sub": "keep",
            "env GOMAXPROCS=1 go build -p 1 .": "keep",
        }
        for segment, expected in cases.items():
            with self.subTest(segment=segment):
                self.assertEqual(classify_segment(segment, REPO), expected)

    def test_chains(self) -> None:
        """A chain keeps its build parts; harmless leftovers drop."""
        kept, dropped = replayable(
            "apt-get update && apt-get install -y gcc && cd build && make",
            REPO,
        )
        self.assertEqual(kept, "cd build && make")
        self.assertEqual(len(dropped), 2)
        kept, dropped = replayable(
            "make && cd /tmp && git clone x && make && make install", REPO
        )
        self.assertEqual(kept, "make")
        self.assertEqual(dropped[0], "cd /tmp")
        self.assertEqual(len(dropped), 4)
        self.assertEqual(
            replayable(f"cd {REPO} && make install", REPO)[0], None
        )
        self.assertIsNone(replayable('echo "No dependencies"', REPO)[0])

    def test_rewrite_keeps_other_repositories(self) -> None:
        """Only the baseline tree path changes, never a longer name."""
        command = (
            f"cmake -S {REPO} -B {REPO}/build "
            f"-DX={REPO}-ng -DY='{REPO}' && cd {REPO}"
        )
        dest = f"{ROOT}/v1-tune"
        self.assertEqual(
            rewrite_for_tree(command, REPO, dest),
            f"cmake -S {dest} -B {dest}/build -DX={REPO}-ng "
            f"-DY='{dest}' && cd {dest}",
        )

    def test_replay_uses_the_executed_and_retried_commands(self) -> None:
        """The builder's rewrite and a retry stored under its own key."""
        state = _state()
        bootstrap = "autoreconf -fi && ./configure"
        state.build_plan.phases[1].commands = ["./configure", "go build ."]
        state.cache_command_result(
            "./configure", CommandResult(bootstrap, 0, "", "", 1.0)
        )
        failed = CommandResult("go build -buildvcs=false .", 1, "", "x", 1)
        state.cache_command_result("go build .", failed)
        state.cache_command_result(
            failed.command, CommandResult(failed.command, 0, "", "", 1.0)
        )
        commands, dropped = replay_commands(state)
        self.assertEqual(commands, [bootstrap, "go build -buildvcs=false ."])
        self.assertEqual(dropped, ["apt-get install -y cmake"])

    def test_variant_env(self) -> None:
        """C variants get the wrappers; the tree token is filled in."""
        variant = Variant(
            "v6-pgo",
            "pgo",
            "c",
            "clang-19",
            flags=["-O3", f"-fprofile-prefix-path={TREE_TOKEN}"],
            env={"CCACHE_DISABLE": "1"},
        )
        dest = f"{ROOT}/v6-pgo"
        env = variant_env(variant, dest)
        self.assertEqual(env["CC"], f"{dest}/.atesor/cc")
        self.assertEqual(env["ATESOR_REAL_CXX"], "clang++-19")
        self.assertEqual(
            env["ATESOR_CC_FLAGS"],
            f"-O3 -fprofile-prefix-path={dest}",
        )
        self.assertEqual(
            env["MAKEFLAGS"], f"CC={dest}/.atesor/cc CXX={dest}/.atesor/cxx"
        )
        self.assertEqual(env["CCACHE_DISABLE"], "1")
        go = variant_env(
            Variant("g1", "go", "go", env={"GORISCV64": "rva22u64"}), dest
        )
        self.assertEqual(go, {"GORISCV64": "rva22u64"})


class TestPrograms(unittest.TestCase):
    """The in-sandbox programs run for real on this host."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _run(self, program, spec, check=True, env=None):
        done = subprocess.run(
            [sys.executable, "-c", program, json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if check:
            self.assertEqual(done.returncode, 0, done.stderr)
            line = done.stdout.strip().splitlines()[-1]
            self.assertTrue(line.startswith(optimize._MARKER))
            return json.loads(line[len(optimize._MARKER) :])
        return done

    def _write(self, rel, data=b"text\n"):
        path = os.path.join(self.dir, "repo", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def _git(self, *args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                *args,
            ],
            cwd=os.path.join(self.dir, "repo"),
            check=True,
            capture_output=True,
        )

    def test_programs_parse(self) -> None:
        import ast

        for program in (
            optimize._PREPARE_PROGRAM,
            optimize._INSPECT_PROGRAM,
            optimize._FINISH_PROGRAM,
        ):
            ast.parse(program)

    def test_prepare_removes_outputs_and_keeps_sources(self) -> None:
        """Untracked compiled outputs go; tracked files and text stay."""
        elf = b"\x7fELF\x02\x01\x01" + b"\0" * 32
        self._write("src.c")
        self._write("Makefile")
        self._write("fixtures/prebuilt.o", elf)
        self._write("docs/build/index.md")
        self._write("sub/a.c")
        self._git("init", "-q")
        self._git("add", ".")
        self._git("commit", "-q", "-m", "init")
        self._write("build/out.o", elf)
        self._write("tools/build/notes.txt")
        self._write("cmake-out/CMakeCache.txt")
        self._write("cmake-out/x.o", elf)
        self._write("meson-out/meson-private/coredata.dat")
        self._write("sub/CMakeCache.txt")
        self._write("sub/CMakeFiles/progress.marks")
        self._write("lib.lo")
        self._write("app", elf)
        self._write("libx.a", b"!<arch>\nrest")
        self._write("notes.txt")
        os.symlink("src.c", os.path.join(self.dir, "repo", "link.o"))
        root = os.path.join(self.dir, "repo.opt")
        dest = os.path.join(root, "v1")
        removed = self._run(
            optimize._PREPARE_PROGRAM,
            {
                "src": os.path.join(self.dir, "repo"),
                "dest": dest,
                "root": root,
                "build_dirs": list(optimize.BUILD_DIRS),
            },
        )
        kept = (
            "src.c",
            "Makefile",
            "fixtures/prebuilt.o",
            "docs/build/index.md",
            "sub/a.c",
            "tools/build/notes.txt",
            "notes.txt",
            "link.o",
            ".git/HEAD",
        )
        gone = (
            "build",
            "cmake-out",
            "meson-out",
            "sub/CMakeCache.txt",
            "sub/CMakeFiles",
            "lib.lo",
            "app",
            "libx.a",
        )
        for rel in kept:
            self.assertTrue(os.path.lexists(os.path.join(dest, rel)), rel)
        for rel in gone:
            self.assertFalse(os.path.lexists(os.path.join(dest, rel)), rel)
        self.assertIn("build", removed["dirs"])
        self.assertTrue(
            os.path.isfile(os.path.join(self.dir, "repo", "app")),
            "the baseline tree must not change",
        )
        wrapper = os.path.join(dest, ".atesor", "cc")
        self.assertTrue(os.stat(wrapper).st_mode & stat.S_IXUSR)
        log = os.path.join(self.dir, "cc.log")
        done = subprocess.run(
            [wrapper, "-c", "x.c"],
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ["PATH"],
                "ATESOR_REAL_CC": "echo",
                "ATESOR_CC_FLAGS": "-O3 -march=rv64gc_zba",
                "ATESOR_CC_LOG": log,
            },
        )
        self.assertEqual(done.stdout, "-c x.c -O3 -march=rv64gc_zba\n")
        with open(log) as handle:
            self.assertEqual(handle.read(), "-c x.c\n")

    def test_prepare_refuses_paths_outside_the_root(self) -> None:
        """The destination must sit directly in a .opt root."""
        root = os.path.join(self.dir, "repo.opt")
        for dest, root_value in (
            (os.path.join(self.dir, "elsewhere"), root),
            (os.path.join(root, "a", "b"), root),
            (os.path.join(self.dir, "x", "v1"), os.path.join(self.dir, "x")),
        ):
            done = self._run(
                optimize._PREPARE_PROGRAM,
                {
                    "src": self.dir,
                    "dest": dest,
                    "root": root_value,
                    "build_dirs": [],
                },
                check=False,
            )
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("unsafe variant path", done.stderr)

    def test_decoder_matches_the_assembler(self) -> None:
        """Every reference encoding gets its extension class."""
        namespace = _inspect_namespace()
        for expected, name, word in ENCODINGS:
            data = bytes.fromhex(word)
            if len(data) == 2:
                got = namespace["classify16"](data[0] | data[1] << 8)
            else:
                got = namespace["classify32"](int.from_bytes(data, "little"))
            with self.subTest(instruction=name):
                self.assertEqual(got, expected)

    def test_inspect_reads_elf_facts(self) -> None:
        """Counts, attributes, hashes and the wrapper log come back."""
        tree = os.path.join(self.dir, "tree")
        os.makedirs(os.path.join(tree, "bin"))
        image = _elf(
            _code(
                "sh3add a0, a1, a2",
                "andn a0, a1, a2",
                "bseti a0, a1, 40",
                "c.zext.b a0",
                "add a0, a1, a2",
            )
        )
        with open(os.path.join(tree, "bin", "tool"), "wb") as handle:
            handle.write(image)
        with open(os.path.join(tree, "README"), "w") as handle:
            handle.write("text")
        log = os.path.join(self.dir, "cc.log")
        with open(log, "w") as handle:
            handle.write("a\nb\nc\n")
        facts = self._run(
            optimize._INSPECT_PROGRAM,
            {
                "tree": tree,
                "artifacts": ["bin/tool", "README", "gone"],
                "log": log,
                "go": False,
                "max_scans": 1,
                "max_bytes": 1 << 20,
            },
        )
        self.assertEqual(facts["cc_calls"], 3)
        tool = facts["artifacts"]["bin/tool"]
        self.assertEqual(tool["machine"], 243)
        self.assertEqual(tool["arch"], "rv64i2p1_m2p0_zba1p0")
        self.assertEqual(tool["unaligned"], 1)
        self.assertEqual(
            tool["counts"], {"zba": 1, "zbb": 1, "zbs": 1, "zcb": 1}
        )
        self.assertEqual(tool["sha256"], hashlib.sha256(image).hexdigest())
        code = _code(
            "sh3add a0, a1, a2",
            "andn a0, a1, a2",
            "bseti a0, a1, 40",
            "c.zext.b a0",
            "add a0, a1, a2",
        )
        self.assertEqual(tool["code_sha256"], hashlib.sha256(code).hexdigest())
        self.assertFalse(facts["artifacts"]["README"]["elf"])
        self.assertFalse(facts["artifacts"]["gone"]["exists"])
        skipped = self._run(
            optimize._INSPECT_PROGRAM,
            {
                "tree": tree,
                "artifacts": ["bin/tool"],
                "log": None,
                "go": False,
                "max_scans": 0,
                "max_bytes": 1 << 20,
            },
        )
        self.assertIsNone(skipped["cc_calls"])
        self.assertIsNone(skipped["artifacts"]["bin/tool"]["counts"])
        self.assertNotIn("partial", skipped["artifacts"]["bin/tool"])

    def test_inspect_reads_archive_facts(self) -> None:
        """A static archive: its objects' machine, ISA counts and code."""
        tree = os.path.join(self.dir, "tree")
        os.makedirs(os.path.join(tree, "lib"))
        first = _code("sh3add a0, a1, a2", "add a0, a1, a2")
        second = _code("andn a0, a1, a2")
        x86 = bytearray(_elf(second))
        x86[18:20] = struct.pack("<H", 62)
        archives = {
            "lib/libdemo.a": _ar(
                [
                    ("/", b"symbol table"),
                    ("a.o/", _elf(first)),
                    ("#1/8", b"long.o\0\0" + _elf(second)),
                ]
            ),
            "lib/libx86.a": _ar([("a.o/", bytes(x86))]),
            # LTO intermediate code only: no executable bytes.
            "lib/liblto.a": _ar([("lto.o/", _elf(b""))]),
        }
        for rel, data in archives.items():
            with open(os.path.join(tree, rel), "wb") as handle:
                handle.write(data)
        facts = self._run(
            optimize._INSPECT_PROGRAM,
            {
                "tree": tree,
                "artifacts": list(archives),
                "log": None,
                "go": False,
                "max_scans": 3,
                "max_bytes": 1 << 20,
            },
        )
        demo = facts["artifacts"]["lib/libdemo.a"]
        self.assertTrue(demo["archive"])
        self.assertFalse(demo["elf"])
        self.assertEqual(demo["members"], 2)
        self.assertEqual(demo["machine"], 243)
        self.assertEqual(demo["arch"], "rv64i2p1_m2p0_zba1p0")
        self.assertEqual(demo["counts"], {"zba": 1, "zbb": 1})
        self.assertEqual(
            demo["code_sha256"], hashlib.sha256(first + second).hexdigest()
        )
        self.assertEqual(facts["artifacts"]["lib/libx86.a"]["machine"], 62)
        lto = facts["artifacts"]["lib/liblto.a"]
        self.assertEqual(lto["members"], 1)
        self.assertNotIn("code_sha256", lto)

    def test_inspect_returns_the_log_of_the_baseline(self) -> None:
        """read_log brings the compiler calls back; else they stay."""
        tree = os.path.join(self.dir, "tree")
        os.makedirs(tree)
        log = os.path.join(self.dir, "cc.log")
        with open(log, "w") as handle:
            handle.write("-O2 -c a.c -o a.o\n-O2 -c b.c -o b.o\n")
        spec = {
            "tree": tree,
            "artifacts": [],
            "log": log,
            "go": False,
            "max_scans": 0,
            "max_bytes": 0,
        }
        facts = self._run(optimize._INSPECT_PROGRAM, dict(spec, read_log=True))
        self.assertEqual(
            facts["log_lines"], ["-O2 -c a.c -o a.o", "-O2 -c b.c -o b.o"]
        )
        self.assertEqual(facts["cc_calls"], 2)
        self.assertNotIn(
            "log_lines", self._run(optimize._INSPECT_PROGRAM, spec)
        )

    def test_inspect_finds_artifacts_in_another_hash_directory(self) -> None:
        """lz4 names its output directory after a hash of its flags."""
        tree = os.path.join(self.dir, "tree")
        old = os.path.join(tree, "lib", "cachedObjs", "a" * 32)
        new = os.path.join(tree, "lib", "cachedObjs", "b" * 32)
        os.makedirs(old)
        os.makedirs(new)
        with open(os.path.join(new, "liblz4.a"), "wb") as handle:
            handle.write(b"!<arch>\n")
        facts = self._run(
            optimize._INSPECT_PROGRAM,
            {
                "tree": tree,
                "artifacts": [
                    f"lib/cachedObjs/{'c' * 32}/liblz4.a",
                    "lib/plain.a",
                ],
                "log": None,
                "go": False,
                "max_scans": 0,
                "max_bytes": 0,
            },
        )
        found = facts["artifacts"][f"lib/cachedObjs/{'c' * 32}/liblz4.a"]
        self.assertTrue(found["exists"])
        self.assertEqual(found["path"], f"lib/cachedObjs/{'b' * 32}/liblz4.a")
        self.assertFalse(facts["artifacts"]["lib/plain.a"]["exists"])
        self.assertNotIn("path", facts["artifacts"]["lib/plain.a"])

    def test_code_hash_ignores_non_code_sections(self) -> None:
        """Two files with the same code but other attributes match."""
        tree = os.path.join(self.dir, "tree")
        os.makedirs(tree)
        code = _code("sh3add a0, a1, a2", "add a0, a1, a2")
        for name, arch in (("a", "rv64i2p1_zba1p0"), ("b", "rv64i2p1_m2p0")):
            with open(os.path.join(tree, name), "wb") as handle:
                handle.write(_elf(code, arch=arch))
        facts = self._run(
            optimize._INSPECT_PROGRAM,
            {
                "tree": tree,
                "artifacts": ["a", "b"],
                "log": None,
                "go": False,
                "max_scans": 0,
                "max_bytes": 0,
            },
        )
        first, second = facts["artifacts"]["a"], facts["artifacts"]["b"]
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["code_sha256"], second["code_sha256"])

    def test_finish_copies_and_removes(self) -> None:
        """Accepted artifacts are copied; only named entries go."""
        root = os.path.join(self.dir, "r.opt")
        for rel in ("v1/bin/tool", "v2/x", "inputs/in", "accepted/old"):
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                handle.write(rel)
        os.makedirs(os.path.join(self.dir, "escape"))
        result = self._run(
            optimize._FINISH_PROGRAM,
            {
                "root": root,
                "keep_tree": os.path.join(root, "v1"),
                "artifacts": ["bin/tool", "missing"],
                "remove": ["v1", "v2", "inputs", "../escape", "accepted"],
            },
        )
        self.assertEqual(result["copied"], ["bin/tool"])
        self.assertEqual(os.listdir(root), ["accepted"])
        self.assertEqual(os.listdir(os.path.join(root, "accepted")), ["bin"])
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "escape")))
        self._run(optimize._FINISH_PROGRAM, {"root": root, "reset": True})
        self.assertFalse(os.path.exists(root))
        os.makedirs(os.path.join(root, "v3"))
        self._run(optimize._FINISH_PROGRAM, {"root": root, "remove": ["v3"]})
        self.assertFalse(os.path.exists(root), "an empty root is removed")
        done = self._run(
            optimize._FINISH_PROGRAM,
            {"root": os.path.join(self.dir, "repo"), "reset": True},
            check=False,
        )
        self.assertNotEqual(done.returncode, 0)


def _facts(sha="new", counts=None, cc_calls=4, **extra):
    info = {
        "exists": True,
        "elf": True,
        "machine": 243,
        "sha256": sha,
        "counts": counts or {},
    }
    info.update(extra)
    return {"cc_calls": cc_calls, "artifacts": {"bin/tool": info}}


class TestGates(unittest.TestCase):
    """Gates 1 and 2 decide what reaches a measurement."""

    def setUp(self) -> None:
        self.variant = Variant("v2-isa", "isa", "c", "gcc", ["-O2"])
        self.profile = _profile()

    def test_gate_one_failures(self) -> None:
        """Missing, foreign or unflagged artifacts fail gate 1."""
        missing = {"cc_calls": 1, "artifacts": {"x": {"exists": False}}}
        self.assertEqual(
            gate_build_facts(missing, self.variant, self.profile)[0],
            "failed",
        )
        foreign = _facts(machine=62)
        self.assertIn(
            "not riscv64",
            gate_build_facts(foreign, self.variant, self.profile)[1],
        )
        silent = _facts(cc_calls=0)
        self.assertIn(
            "did not reach",
            gate_build_facts(silent, self.variant, self.profile)[1],
        )
        self.assertEqual(
            gate_build_facts({"artifacts": {}}, self.variant, self.profile)[0],
            "failed",
        )

    def test_a_helper_the_measurement_never_runs_only_warns(self) -> None:
        """A missing artifact outside ``required`` does not fail a tune."""
        facts = _facts()
        rel = next(iter(facts["artifacts"]))
        facts["artifacts"]["build/simd/simdcoverage"] = {"exists": False}
        # Without a required list every artifact must exist: that is
        # what the rebuild of the verified build gets.
        outcome, reason, _ = gate_build_facts(
            facts, self.variant, self.profile
        )
        self.assertEqual(outcome, "failed")
        self.assertIn("simdcoverage was not built", reason)
        # The measurement links `rel` and never runs simdcoverage.
        outcome, reason, warnings = gate_build_facts(
            facts, self.variant, self.profile, None, [rel]
        )
        self.assertEqual(outcome, "passed")
        self.assertTrue(
            any(
                "simdcoverage was not built" in text
                and "does not fail the tune" in text
                for text in warnings
            )
        )
        # A library the measurement does link still fails the gate.
        facts["artifacts"][rel] = {"exists": False}
        outcome, reason, _ = gate_build_facts(
            facts, self.variant, self.profile, None, [rel]
        )
        self.assertEqual(outcome, "failed")
        self.assertIn(f"{rel} was not built", reason)

    def test_archives_count_in_gate_1(self) -> None:
        """A static archive is checked for riscv64 and compared by code."""

        def archive(machine=243, code="c1"):
            info = {
                "exists": True,
                "elf": False,
                "archive": True,
                "machine": machine,
                "sha256": "file",
                "counts": {},
            }
            if code:
                info["code_sha256"] = code
            return {"cc_calls": 3, "artifacts": {"lib/libopus.a": info}}

        self.assertIn(
            "not riscv64",
            gate_build_facts(archive(machine=62), self.variant, self.profile)[
                1
            ],
        )
        self.assertEqual(
            gate_build_facts(archive(), self.variant, self.profile, archive())[
                0
            ],
            "no effect",
        )
        self.assertEqual(
            gate_build_facts(
                archive(code="c2"), self.variant, self.profile, archive()
            )[0],
            "passed",
        )
        lto = archive(code=None)
        self.assertEqual(optimize._code_artifacts(lto), [])
        self.assertEqual(
            gate_build_facts(lto, self.variant, self.profile, lto)[0], "passed"
        )
        self.assertEqual(optimize._same_code(archive(), archive()), (1, 1))
        self.assertEqual(optimize._same_code(lto, lto), (0, 0))

    def test_go_settings_must_reach_the_binary(self) -> None:
        """GORISCV64 and -pgo must show in the build information."""
        go = Variant("g2-pgo", "pgo", "go", isa="rva22u64", pgo="use")
        level = _facts(goriscv64="rva20u64", cc_calls=None)
        self.assertIn(
            "GORISCV64=rva22u64", gate_build_facts(level, go, self.profile)[1]
        )
        no_pgo = _facts(goriscv64="rva22u64", cc_calls=None)
        self.assertIn("-pgo", gate_build_facts(no_pgo, go, self.profile)[1])
        good = _facts(
            goriscv64="rva22u64", pgo="/p/default.pgo", cc_calls=None
        )
        self.assertEqual(gate_build_facts(good, go, self.profile)[0], "passed")

    def test_no_effect_and_warnings(self) -> None:
        """An identical artifact has no effect; new risks warn."""
        baseline = _facts(sha="same", counts={"zbs": 2})
        self.assertEqual(
            gate_build_facts(
                _facts(sha="same"), self.variant, self.profile, baseline
            )[0],
            "no effect",
        )
        risky = _facts(counts={"zbs": 1, "v": 3, "zba": 9}, unaligned=1)
        outcome, _, warnings = gate_build_facts(
            risky, self.variant, self.profile, baseline
        )
        self.assertEqual(outcome, "passed")
        text = " ".join(warnings)
        self.assertIn("contains v instructions", text)
        self.assertNotIn("zbs", text, "the baseline had Zbs already")
        self.assertIn("misaligned", text)

    def test_same_code_is_no_effect(self) -> None:
        """A different file with the same code had no effect."""
        baseline = _facts(sha="file-a", code_sha256="code")
        same_code = _facts(sha="file-b", code_sha256="code")
        self.assertEqual(
            gate_build_facts(same_code, self.variant, self.profile, baseline)[
                0
            ],
            "no effect",
        )
        other = _facts(sha="file-b", code_sha256="other")
        self.assertEqual(
            gate_build_facts(other, self.variant, self.profile, baseline)[0],
            "passed",
        )

    def test_gate_two(self) -> None:
        """The variant must pass only what the baseline passes."""
        passed = {"status": "passed"}
        self.assertEqual(gate_tests({"status": "failed"}, None)[0], True)
        self.assertIn("not decisive", gate_tests(None, None)[1])
        self.assertEqual(gate_tests(passed, {"status": "passed"})[0], True)
        self.assertEqual(
            gate_tests(passed, {"status": "timeout"}),
            (False, "the package tests timeout"),
        )
        self.assertEqual(gate_tests(passed, None)[0], False)


def _ok(stdout=""):
    return CommandResult("cmd", 0, stdout, "", 0.1)


def _measured(ratio, verdict):
    return {
        "status": "measured",
        "reason": verdict,
        "warnings": [],
        "cv": 0.01,
        "mde": 0.009,
        "rounds": 20,
        "board_pre": {"governor": "performance"},
        "comparison": {
            "ratio": ratio,
            "ci_low": round(ratio - 0.01, 4),
            "ci_high": round(ratio + 0.01, 4),
            "p_value": 0.001,
            "verdict": verdict,
        },
    }


def _program_workload(**overrides):
    data = dict(
        mode="program",
        command="{bin} < {input} > {output}",
        binary="build/minigzip",
        input_kind="binary",
        input_mb=8,
        description="compress the input",
        run_seconds=1.0,
        origin="agent",
    )
    data.update(overrides)
    return bench.Workload(**data)


def _harness_workload():
    return bench.Workload(
        mode="harness",
        command="{bin} {input} {iters}",
        binary=".atesor/bench/harness",
        input_kind="binary",
        input_mb=8,
        description="deflate and inflate the input",
        run_seconds=1.9,
        iterations=14,
        check_prefix="ATESOR-CHECK",
        files={".atesor/bench/harness.c": "int main(void) { return 0; }"},
        build=['mkdir -p .atesor/bench && "$CC" -O2 harness.c'],
        source_language="c",
        origin="agent",
    )


# The baseline tree's compiler calls: -O2, no LTO.
_BASE_LOG = ["-O2 -c /w/zlib.opt/base/adler32.c -o adler32.o"]


class _Sandbox:
    """Fakes every sandbox call that run_optimization makes."""

    def __init__(
        self,
        profile,
        native=True,
        ratios=None,
        failing=(),
        busy=(),
        workload="program",
        notes=(),
        log_lines=None,
        health=None,
        crashes=(),
    ):
        self.profile = profile
        self.health = health or {
            "bad": {"0": 0, "1": 0, "2": 0, "3": 0},
            "rounds": 10,
            "size_mb": 8,
            "seconds": 6.1,
        }
        self.health_calls = []
        self.native = native
        self.ratios = ratios or {}
        self.failing = failing
        self.busy = busy
        self.workload = (
            _program_workload() if workload == "program" else workload
        )
        self.notes = list(notes)
        self.crashes = list(crashes)
        self.log_lines = _BASE_LOG if log_lines is None else log_lines
        self.builds = {}
        self.installed = []
        self.install_paths = {}
        self.jobs = {}
        self.bases = {}
        self.programs = []
        self.measured = []
        self.checked = []
        self.trained = []
        self.contexts = []

    @staticmethod
    def vid(path):
        return path.rstrip("/").rsplit("/", 1)[-1]

    def build(self, commands, repo_path, dest, env, timeout):
        self.builds[self.vid(dest)] = dict(env)
        if self.vid(dest) in self.failing:
            return "`make` exited 2: error"
        return None

    def inspect(self, tree, artifacts, log, go, kind, read_log=False):
        # The baseline copy rebuilds the verified code.
        sha = "base" if tree == REPO else self.vid(tree)
        facts = {
            "cc_calls": 7 if log else None,
            "artifacts": {
                rel: {
                    "exists": True,
                    "elf": not rel.endswith(".a"),
                    "machine": 243,
                    "sha256": sha,
                    "counts": {},
                }
                for rel in artifacts
            },
        }
        if read_log:
            facts["log_lines"] = list(self.log_lines)
        return facts, ""

    def integrity(self, cpus):
        self.health_calls.append(list(cpus))
        return self.health, ""

    def choose(self, ctx):
        self.contexts.append(ctx)
        ctx.crashes.extend(self.crashes)
        workload = self.workload
        return (
            bench.Workload(**workload.to_dict()) if workload else None,
            list(self.notes),
        )

    def install(self, workload, tree, env, timeout, paths=None):
        self.installed.append(self.vid(tree))
        self.install_paths[self.vid(tree)] = paths
        return None

    def program(self, program, spec, timeout):
        self.programs.append(spec)
        return {"copied": list(spec.get("artifacts") or [])}, ""

    def measure(self, base_job, variant_job, workload, pin, timeout):
        vid = self.vid(variant_job["cwd"])
        self.measured.append((vid, pin))
        self.bases[vid] = base_job["cwd"]
        self.jobs[vid] = (base_job, variant_job)
        ratio, verdict = self.ratios.get(vid, (1.0, "no change"))
        return _measured(ratio, verdict)

    def check_once(self, base_job, variant_job, workload, extra_env=None):
        self.checked.append((self.vid(variant_job["cwd"]), extra_env))
        return {"status": "checked", "reason": "outputs match"}

    def run_jobs(self, spec, timeout, extra_env=None):
        self.trained.append((spec, extra_env))
        run = {
            "job": "B",
            "phase": "round",
            "round": 0,
            "wall_ns": 1_000_000_000,
            "rc": 0,
            "sha": "x",
        }
        return {"runs": [run, dict(run, round=1)], "error": None}

    def __enter__(self):
        patches = [
            mock.patch.object(optimize, "is_native", return_value=self.native),
            mock.patch.object(
                optimize, "probe_machine", return_value=(self.profile, None)
            ),
            mock.patch.object(optimize, "prepare_tree", return_value=None),
            mock.patch.object(
                optimize, "build_variant", side_effect=self.build
            ),
            mock.patch.object(
                optimize, "inspect_tree", side_effect=self.inspect
            ),
            mock.patch.object(
                optimize, "_run_program", side_effect=self.program
            ),
            mock.patch.object(optimize, "execute_command", return_value=_ok()),
            mock.patch.object(
                optimize, "_other_containers", return_value=list(self.busy)
            ),
            mock.patch.object(
                optimize, "check_integrity", side_effect=self.integrity
            ),
            mock.patch.object(
                workload_module, "choose_workload", side_effect=self.choose
            ),
            mock.patch.object(
                workload_module, "install_workload", side_effect=self.install
            ),
            mock.patch.object(bench, "measure", side_effect=self.measure),
            mock.patch.object(
                bench, "check_once", side_effect=self.check_once
            ),
            mock.patch.object(bench, "run_jobs", side_effect=self.run_jobs),
            mock.patch.object(bench, "write_input", return_value=None),
        ]
        self._patches = patches
        for patch in patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        return False


def _by_id(result):
    return {r["variant"]["vid"]: r for r in result["variants"]}


class TestRunOptimization(unittest.TestCase):
    """The whole flow with a fake sandbox."""

    def test_a_machine_that_gives_wrong_data_stops_first(self) -> None:
        """The machine check fails: nothing is built or timed."""
        sandbox = _Sandbox(
            _profile(),
            health={
                "bad": {"0": 1, "1": 2, "2": 0, "3": 0},
                "rounds": 10,
                "size_mb": 8,
                "seconds": 6.4,
            },
        )
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        self.assertEqual(sandbox.health_calls, [[0, 1, 2, 3]])
        self.assertEqual(result["status"], "not measured")
        self.assertEqual(result["headline"], "NO: no tune could be timed")
        self.assertTrue(
            result["reason"].startswith(
                "the machine gives wrong results: 3 of 40 zlib round trips"
            )
        )
        self.assertEqual(sandbox.builds, {})
        self.assertEqual(sandbox.contexts, [])
        self.assertIn(
            "Health   : 3 of 40 zlib round trips gave wrong data "
            "(CPU 0: 1, CPU 1: 2)",
            lines,
        )
        self.assertTrue(
            lines[-1].startswith(
                "           fix: make the " "machine stable first"
            )
        )
        crashed = _Sandbox(
            _profile(),
            health={
                "bad": {"0": -1},
                "rounds": 10,
                "size_mb": 8,
                "seconds": 1.0,
            },
        )
        with crashed:
            result = run_optimization(_state())
        self.assertIn("(CPU 0: crashed)", result["reason"])

    def test_machine_check_passes_and_is_recorded(self) -> None:
        """A good check prints one line and lets the run go on."""
        sandbox = _Sandbox(_profile())
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        self.assertIn(
            "Health   : 40 zlib round trips on 4 CPUs gave the "
            "same data (6.1 s)",
            lines,
        )
        self.assertEqual(result["machine_check"]["rounds"], 10)
        self.assertIn("tune1", sandbox.builds)
        qemu = _Sandbox(
            _profile(
                kind="qemu", core_name=None, emulated_cpu="rv64i,zba=true"
            ),
            native=False,
        )
        with qemu:
            run_optimization(_state())
        self.assertEqual(qemu.health_calls, [], "QEMU has no machine check")

    def test_unverified_port_is_skipped(self) -> None:
        lines = []
        result = run_optimization(
            _state(verified=False), progress=lines.append
        )
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            lines,
            [
                "Result   : NO: performance mode was skipped",
                "           why: the port is not verified, so there is no "
                "baseline",
            ],
        )

    def test_probe_failure_is_an_error(self) -> None:
        """A failed probe stops the run with its reason."""
        with (
            mock.patch.object(optimize, "is_native", return_value=True),
            mock.patch.object(
                optimize,
                "probe_machine",
                return_value=(None, "exit 127: boom"),
            ),
        ):
            result = run_optimization(_state())
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["reason"])

    def test_native_flow_accepts_a_confirmed_tune(self) -> None:
        """Tune-3 wins twice; PGO builds on it; Tune-3 is accepted."""
        state = _state()
        tests_seen = []

        def run_tests(tree, env):
            tests_seen.append(tree.rsplit("/", 1)[-1])
            if tree.endswith("/tune2"):
                return {"status": "failed"}
            return {"status": "passed"}

        sandbox = _Sandbox(
            _profile(),
            ratios={
                "tune1": (0.99, "no change"),
                "tune3": (0.90, "faster"),
                "tune3-layout2": (0.91, "faster"),
                "tune4": (1.05, "slower"),
                "tune6": (0.95, "no change"),
            },
        )
        lines = []
        with sandbox:
            result = run_optimization(
                state, run_tests=run_tests, progress=lines.append
            )
        verdicts = {vid: r["verdict"] for vid, r in _by_id(result).items()}
        self.assertEqual(
            verdicts,
            {
                "tune1": "no change",
                "tune2": "gate 2 failed",
                "tune3": "faster",
                "tune4": "slower",
                "tune5": "no change",
                "tune6": "no change",
            },
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["accepted"], "tune3")
        self.assertEqual(result["accepted_label"], "Tune-3")
        self.assertEqual(result["accepted_ratio"], 0.90)
        self.assertEqual(result["accepted_dir"], f"{ROOT}/accepted")
        self.assertEqual(
            result["accepted_artifacts"],
            ["build/minigzip", "build/libz.so.1", "build/libz.a"],
        )
        self.assertEqual(result["package"]["kind"], "library+programs")
        self.assertEqual(
            result["baseline"],
            {
                "build_seconds": result["baseline"]["build_seconds"],
                "same_code": 2,
                "artifacts": 2,
            },
        )
        self.assertEqual(state.machine_profile["core_name"], "sifive-p550")
        # Every tune is compared with the rebuilt copy, not the port.
        self.assertEqual(sandbox.bases["tune1"], f"{ROOT}/base")
        self.assertEqual(
            sandbox.bases["tune3-layout2"], f"{ROOT}/base-layout2"
        )
        self.assertEqual(sandbox.contexts[0].tree, f"{ROOT}/base")
        # The copy gets no flag at all, so it is the verified build.
        self.assertEqual(sandbox.builds["base"]["ATESOR_CC_FLAGS"], "")
        self.assertNotIn("tune6-train", tests_seen)
        self.assertNotIn("base", tests_seen)
        self.assertIn("tune3-layout2", tests_seen)
        self.assertTrue(all(pin == 3 for _, pin in sandbox.measured))
        flags = sandbox.builds["tune3"]["ATESOR_CC_FLAGS"].split()
        self.assertEqual(flags[-2:], ["-march=rv64gc_zba_zbb", "-O3"])
        train = sandbox.builds["tune6-train"]
        self.assertIn(
            f"-fprofile-generate={ROOT}/pgo-tune6", train["ATESOR_CC_FLAGS"]
        )
        self.assertIn(
            f"-fprofile-prefix-path={ROOT}/tune6-train",
            train["ATESOR_CC_FLAGS"],
        )
        self.assertEqual(train["CCACHE_DISABLE"], "1")
        use = sandbox.builds["tune6"]["ATESOR_CC_FLAGS"]
        self.assertIn(f"-fprofile-prefix-path={ROOT}/tune6", use)
        self.assertIn("-Wno-coverage-mismatch", use)
        self.assertIn(
            "-falign-functions=64",
            sandbox.builds["tune3-layout2"]["ATESOR_CC_FLAGS"],
        )
        finish = sandbox.programs[-1]
        self.assertEqual(finish["keep_tree"], f"{ROOT}/tune3")
        # Static archives ship too, not only ELF files.
        self.assertEqual(
            finish["artifacts"],
            ["build/minigzip", "build/libz.so.1", "build/libz.a"],
        )
        for name in ("base", "tune3", "pgo-tune6", "inputs"):
            self.assertIn(name, finish["remove"])
        self.assertTrue(sandbox.programs[0].get("reset"))
        self.assertNotIn("env", result["variants"][0])
        self.assertNotIn("facts", result["variants"][0])
        self.assertIn(
            "is 10.0% faster than the verified build", result["reason"]
        )
        self.assertTrue(
            result["headline"].startswith(
                "YES: Tune-3 (GCC -mtune=sifive-p550"
            )
        )
        expected = [
            "Machine  : native, rv64gc+zba+zbb, core sifive-p550, "
            "misaligned access slow",
            "Package  : C/C++ library and programs (cmake): "
            "build/libz.so.1, build/libz.a, build/minigzip",
            "Baseline : rebuilding the verified build in a copy",
            "Tune-1   : core tuning = -mtune=sifive-p550 -mstrict-align",
            "Tune-2   : machine ISA = -march=rv64gc_zba_zbb, on top of "
            "Tune-1",
            "           package tests failed: the package tests failed",
            "Tune-5   : compiler = clang-19 (the flags of Tune-3)",
            "Tune-6   : profile-guided optimization = PGO, trained on a "
            "second input, on top of Tune-3",
            "Check    : Tune-3 again, in a second code layout "
            "(-falign-functions=64 on both builds)",
        ]
        for line in expected:
            self.assertIn(line, lines)
        self.assertTrue(
            lines[-1].startswith(
                "Result   : YES: Tune-3 (GCC -mtune=sifive-p550"
            )
        )
        self.assertTrue(
            lines[-1].endswith("is 10.0% faster than the verified build")
        )
        self.assertIn(
            "           confirmed: 9.0% faster: time ratio 0.91 "
            "(95% CI 0.9-0.92)",
            lines,
        )
        self.assertIn(
            "           package tests failed: the package tests " "failed",
            lines,
        )

    def test_second_layout_compares_like_with_like(self) -> None:
        """Both builds get the layout flag; the baseline is built once."""
        sandbox = _Sandbox(
            _profile(),
            ratios={
                "tune2": (0.95, "faster"),
                "tune3": (0.90, "faster"),
                "tune3-layout2": (1.0, "no change"),
                "tune2-layout2": (0.96, "faster"),
            },
        )
        with sandbox:
            result = run_optimization(_state())
        base2 = sandbox.builds["base-layout2"]
        self.assertEqual(base2["ATESOR_REAL_CC"], "gcc")
        self.assertEqual(base2["ATESOR_CC_FLAGS"], "-falign-functions=64")
        self.assertEqual(
            sandbox.bases["tune3-layout2"], f"{ROOT}/base-layout2"
        )
        self.assertEqual(
            sandbox.bases["tune2-layout2"], f"{ROOT}/base-layout2"
        )
        self.assertEqual(sandbox.bases["tune2"], f"{ROOT}/base")
        self.assertEqual(
            [p for p in sandbox.programs if "base-layout2" in str(p)][-1][
                "remove"
            ].count("base-layout2"),
            1,
        )
        self.assertEqual(result["accepted"], "tune2")
        records = _by_id(result)
        self.assertEqual(records["tune3"]["verdict"], "not confirmed")
        self.assertIn("on both builds", records["tune3"]["reason"])

    def test_identical_binaries_are_measured_once(self) -> None:
        """Tune-3 with the same code as Tune-2 reuses its result."""
        sandbox = _Sandbox(_profile(), ratios={"tune2": (0.97, "faster")})
        original = sandbox.inspect

        def twins(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            if tree.endswith("/tune3"):
                for info in facts["artifacts"].values():
                    info["sha256"] = "tune2"
            return facts, error

        sandbox.inspect = twins
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        record = _by_id(result)["tune3"]
        self.assertEqual(record["verdict"], "same as tune2")
        self.assertEqual(record["same_as_label"], "Tune-2")
        self.assertIn("result reused (3.0% faster", record["reason"])
        self.assertEqual(record["ratio"], 0.97)
        self.assertNotIn("tune3", [vid for vid, _ in sandbox.measured])
        self.assertTrue(any("same code as Tune-2" in line for line in lines))

    def test_failed_layout_baseline_confirms_nothing(self) -> None:
        sandbox = _Sandbox(
            _profile(),
            ratios={"tune3": (0.9, "faster")},
            failing=("base-layout2",),
        )
        with sandbox:
            result = run_optimization(_state())
        record = _by_id(result)["tune3"]
        self.assertEqual(record["verdict"], "not confirmed")
        self.assertIn("no second-layout baseline", record["reason"])
        self.assertIsNone(result["accepted"])
        self.assertEqual(result["status"], "no gain")

    def test_p_value_text(self) -> None:
        """Tiny p-values never print as p=0.0."""
        self.assertEqual(optimize._p_text(0.0), "p<1e-06")
        self.assertEqual(optimize._p_text(0.000338), "p=0.00034")
        self.assertEqual(optimize._p_text(None), "p=?")

    def test_failed_second_check_accepts_nothing(self) -> None:
        sandbox = _Sandbox(
            _profile(),
            ratios={
                "tune3": (0.90, "faster"),
                "tune3-layout2": (1.0, "no change"),
            },
            failing=("tune4",),
        )
        with sandbox:
            result = run_optimization(_state())
        records = _by_id(result)
        self.assertEqual(records["tune3"]["verdict"], "not confirmed")
        self.assertEqual(records["tune4"]["verdict"], "build failed")
        self.assertIsNone(result["accepted"])
        self.assertIsNone(sandbox.programs[-1]["keep_tree"])
        self.assertEqual(result["status"], "no gain")
        self.assertEqual(
            result["reason"],
            "faster in the first measurement, but not in the second "
            "check: Tune-3",
        )

    def test_qemu_flow_checks_without_timing(self) -> None:
        """Under QEMU the tunes get gates and the QEMU_CPU model."""
        state = _state()
        state.emulate_cpu = "rv64i,zba=true"
        profile = _profile(
            kind="qemu", core_name=None, emulated_cpu="rv64i,zba=true"
        )
        sandbox = _Sandbox(profile, native=False)
        with sandbox:
            result = run_optimization(state)
        self.assertEqual(sandbox.measured, [])
        self.assertTrue(sandbox.checked)
        self.assertTrue(
            all(
                env == {"QEMU_CPU": "rv64i,zba=true"}
                for _, env in sandbox.checked
            )
        )
        self.assertEqual(sandbox.builds["tune1"]["QEMU_CPU"], "rv64i,zba=true")
        self.assertEqual(
            sandbox.contexts[0].qemu_env, {"QEMU_CPU": "rv64i,zba=true"}
        )
        self.assertEqual(
            {r["verdict"] for r in result["variants"]}, {"checked"}
        )
        self.assertEqual(
            [r["variant"]["technique"] for r in result["variants"]],
            [
                "machine ISA",
                "optimization level",
                "link-time optimization",
                "compiler",
                "profile-guided optimization",
            ],
        )
        self.assertIsNone(result["accepted"])
        self.assertEqual(result["status"], "checked")
        self.assertIn("QEMU", result["reason"])

    def test_no_workload_on_native_builds_no_tune(self) -> None:
        """Without a workload nothing can be timed, so nothing builds."""
        sandbox = _Sandbox(
            _profile(),
            workload=None,
            notes=[
                "harness: the optimizer agent gave no usable answer (quota)",
                "tests: the package tests did not pass in the port (skipped)",
            ],
        )
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        self.assertEqual(result["status"], "not measured")
        self.assertIn("no workload, so no tune was built", result["reason"])
        self.assertIn("quota", result["reason"])
        self.assertEqual(list(sandbox.builds), ["base"])
        self.assertEqual(result["variants"], [])
        self.assertIn("base", sandbox.programs[-1]["remove"])
        self.assertTrue(
            any(line.startswith("           none: harness:") for line in lines)
        )
        self.assertFalse(any(line.startswith("Tune-") for line in lines))

    def test_a_gate_1_warning_reaches_the_console_and_the_result(
        self,
    ) -> None:
        """A tune that passes with a note says so; silence would hide it."""
        note = "build/simd/simdcoverage was not built; the measurement"

        def warn(facts, variant, profile, baseline=None, required=None):
            return "passed", "ok", [note]

        lines = []
        sandbox = _Sandbox(_profile())
        with (
            sandbox,
            mock.patch.object(optimize, "gate_build_facts", side_effect=warn),
        ):
            result = run_optimization(_state(), progress=lines.append)
        self.assertTrue(
            any(f"note: {note}" in line for line in lines),
            f"no note line: {lines[-6:]}",
        )
        self.assertIn(note, result["warnings"])
        # One warning, however many tunes repeat it.
        self.assertEqual(result["warnings"].count(note), 1)

    def test_gate_1_requires_the_libraries_and_the_timed_program(
        self,
    ) -> None:
        """The run hands gate 1 the artifacts the measurement needs."""
        seen = []
        real = optimize.gate_build_facts

        def record(facts, variant, profile, baseline=None, required=None):
            seen.append(required)
            return real(facts, variant, profile, baseline, required)

        sandbox = _Sandbox(_profile())
        with (
            sandbox,
            mock.patch.object(
                optimize, "gate_build_facts", side_effect=record
            ),
        ):
            run_optimization(_state())
        # The rebuild of the verified build runs before the workload is
        # known, so it keeps the strict check; every tune gets the list.
        self.assertIsNone(seen[0])
        wanted = ["build/libz.a", "build/libz.so.1", "build/minigzip"]
        self.assertTrue(
            any(req == wanted for req in seen),
            f"no gate call got {wanted}: {seen}",
        )

    def test_a_baseline_crash_is_reported_as_a_port_problem(self) -> None:
        """A crash of the verified build leads the reason and the warnings."""
        crash = (
            "the verified build crashed with SIGILL while running the "
            "workload on this machine: the baseline run crashed with "
            "SIGILL in the round phase: Illegal instruction"
        )
        sandbox = _Sandbox(
            _profile(),
            workload=None,
            crashes=[crash],
            notes=[
                "harness: 3 rounds failed; the last error: " + crash,
            ],
        )
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        self.assertEqual(result["status"], "not measured")
        self.assertTrue(result["reason"].startswith(crash))
        self.assertIn("no tune was built or measured", result["reason"])
        self.assertIn(crash, result["warnings"])
        self.assertTrue(any("port problem:" in line for line in lines))
        self.assertEqual(list(sandbox.builds), ["base"])

    def test_a_workload_without_a_crash_keeps_its_own_reason(self) -> None:
        """No crash: the reason stays the list of workload notes."""
        sandbox = _Sandbox(_profile(), workload=None, notes=["harness: x"])
        with sandbox:
            result = run_optimization(_state())
        self.assertIn("no workload, so no tune was built", result["reason"])
        self.assertEqual(result["warnings"], [])

    def test_qemu_without_workload_still_checks_the_builds(self) -> None:
        """Under QEMU the tunes still build and pass gates 1 and 2."""
        profile = _profile(
            kind="qemu", core_name=None, emulated_cpu="rv64i,zba=true"
        )
        sandbox = _Sandbox(profile, native=False, workload=None)
        with sandbox:
            result = run_optimization(_state())
        self.assertEqual({r["verdict"] for r in result["variants"]}, {"built"})
        self.assertEqual(result["status"], "checked")
        self.assertEqual(sandbox.checked, [])

    def test_tunes_that_the_build_has_are_skipped(self) -> None:
        """A build that compiles everything with -O3 skips that tune."""
        sandbox = _Sandbox(
            _profile(),
            log_lines=[
                f"-O3 -DNDEBUG -c {ROOT}/base/a.c -o a.o",
                "-O3 -DNDEBUG -c b.c -o b.o",
            ],
        )
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        record = _by_id(result)["tune3"]
        self.assertEqual(record["verdict"], "skipped")
        self.assertIn("all 2 sources with -O3 already", record["reason"])
        self.assertNotIn("tune3", sandbox.builds)
        self.assertIn("tune4", sandbox.builds)
        self.assertEqual(result["build_flags"]["levels"], ["-O3"])
        self.assertIn(
            "Tune-3   : optimization level = -O3, on top of " "Tune-2", lines
        )
        # Tune-3 adds nothing, so Tune-4 builds on Tune-2.
        self.assertIn(
            "Tune-4   : link-time optimization = -flto=auto "
            "-ffat-lto-objects, on top of Tune-2",
            lines,
        )
        self.assertIn(
            "Tune-5   : compiler = clang-19 (the flags of Tune-3)", lines
        )

    def test_baseline_that_does_not_rebuild_stops(self) -> None:
        sandbox = _Sandbox(_profile(), failing=("base",))
        with sandbox:
            result = run_optimization(_state())
        self.assertEqual(result["status"], "error")
        self.assertIn("did not rebuild: build failed", result["reason"])
        self.assertEqual(list(sandbox.builds), ["base"])
        self.assertEqual(sandbox.contexts, [])

    def test_baseline_with_other_code_warns(self) -> None:
        """A non-reproducible build: tunes compare with the copy."""
        sandbox = _Sandbox(_profile())
        original = sandbox.inspect

        def drift(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            if tree == REPO:
                for info in facts["artifacts"].values():
                    info["sha256"] = "verified"
            if tree.endswith("/tune1"):
                for info in facts["artifacts"].values():
                    info["sha256"] = "base"
            return facts, error

        sandbox.inspect = drift
        with sandbox:
            result = run_optimization(_state())
        self.assertIn(
            "differs from the verified build in 2 of 2",
            " ".join(result["warnings"]),
        )
        self.assertEqual(result["baseline"]["same_code"], 0)
        self.assertEqual(_by_id(result)["tune1"]["verdict"], "no effect")

    def test_harness_builds_in_every_tree(self) -> None:
        sandbox = _Sandbox(
            _profile(),
            workload=_harness_workload(),
            ratios={
                "tune3": (0.9, "faster"),
                "tune3-layout2": (0.9, "faster"),
            },
        )
        with sandbox:
            result = run_optimization(_state())
        self.assertEqual(result["accepted"], "tune3")
        for vid in (
            "tune1",
            "tune5",
            "tune6-train",
            "tune6",
            "base-layout2",
            "tune3-layout2",
        ):
            self.assertIn(vid, sandbox.installed)
        # The workload choice built it in the baseline copy already.
        self.assertNotIn("base", sandbox.installed)
        self.assertEqual(result["workload"]["mode"], "harness")

    def test_identical_build_has_no_effect(self) -> None:
        """A tune equal to the baseline is never measured."""
        sandbox = _Sandbox(_profile())
        original = sandbox.inspect

        def same(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            if tree.endswith("/tune1"):
                for info in facts["artifacts"].values():
                    info["sha256"] = "base"
            return facts, error

        sandbox.inspect = same
        with sandbox:
            result = run_optimization(_state())
        first = result["variants"][0]
        self.assertEqual(first["verdict"], "no effect")
        self.assertNotIn("tune1", [vid for vid, _ in sandbox.measured])

    def test_missing_baseline_artifacts_are_not_required(self) -> None:
        sandbox = _Sandbox(_profile())
        original = sandbox.inspect
        asked = []

        def partial(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            if tree == REPO:
                facts["artifacts"]["build/libz.so.1"]["exists"] = False
            else:
                asked.append(list(artifacts))
            return facts, error

        sandbox.inspect = partial
        with sandbox:
            result = run_optimization(_state())
        self.assertTrue(asked)
        self.assertTrue(
            all(item == ["build/minigzip", "build/libz.a"] for item in asked)
        )
        self.assertEqual(result["status"], "no gain")

        def nothing(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            for info in facts["artifacts"].values():
                info["exists"] = False
            return facts, error

        sandbox = _Sandbox(_profile())
        sandbox.inspect = nothing
        with sandbox:
            result = run_optimization(_state())
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(sandbox.builds, {})

    def test_other_containers_give_a_warning(self) -> None:
        """A busy machine still measures, and the result says why."""
        with _Sandbox(_profile(), busy=("atesor-ai-sandbox-w1",)):
            result = run_optimization(_state())
        self.assertTrue(
            any(
                "atesor-ai-sandbox-w1" in warning
                for warning in result["warnings"]
            )
        )

    def test_a_failing_baseline_stops_the_timing(self) -> None:
        """Two baseline failures: nothing can be trusted, so stop."""
        sandbox = _Sandbox(_profile())

        def flaky(base_job, variant_job, workload, pin, timeout):
            sandbox.measured.append((sandbox.vid(variant_job["cwd"]), pin))
            return {
                "status": "failed",
                "warnings": [],
                "failed_job": "B",
                "reason": "the baseline run exited 1 in the "
                "calibration phase: Round-trip data "
                "mismatch at iteration 0",
            }

        sandbox.measure = flaky
        lines = []
        with sandbox:
            result = run_optimization(_state(), progress=lines.append)
        verdicts = {vid: r["verdict"] for vid, r in _by_id(result).items()}
        self.assertEqual(verdicts["tune1"], "measurement failed")
        self.assertEqual(verdicts["tune2"], "measurement failed")
        for vid in ("tune3", "tune4", "tune5"):
            self.assertEqual(verdicts[vid], "skipped")
            self.assertNotIn(vid, sandbox.builds)
        self.assertNotIn("tune6", verdicts, "no PGO on a failing baseline")
        self.assertEqual(result["status"], "not measured")
        self.assertTrue(
            result["reason"].startswith(
                "the unchanged baseline failed 2 timing runs"
            )
        )
        self.assertIn("Round-trip data mismatch", result["reason"])
        self.assertTrue(
            any(
                "memory or CPU errors" in warning
                for warning in result["warnings"]
            )
        )
        self.assertIn("Result   : NO: no tune could be timed", lines)
        self.assertTrue(
            lines[-2].startswith(
                "           why: the unchanged baseline failed 2 timing runs"
            )
        )
        self.assertEqual(
            lines[-1],
            "           fix: check the machine (a memory test, a lower CPU "
            "frequency), then run again",
        )
        self.assertEqual(result["headline"], "NO: no tune could be timed")

    def test_each_tree_runs_the_artifact_its_build_made(self) -> None:
        """Hash directories: every job names the file its tree built.

        lz4 puts its programs in a directory named after a hash of the
        flags, so each tree has its own. A wildcard also matched the
        directory of the package tests; the shell then passed two
        programs, and the tune run wrote over the input file.
        """
        verified = "build/cachedObjs/3a297cdcdda21e64139b4c81136d85cf/minigzip"
        state = _state()
        state.curated_artifacts[0]["filepath"] = f"{REPO}/{verified}"
        sandbox = _Sandbox(
            _profile(), workload=_program_workload(binary=verified)
        )
        original = sandbox.inspect

        def tree_rel(tree):
            digest = hashlib.md5(sandbox.vid(tree).encode()).hexdigest()
            return f"build/cachedObjs/{digest}/minigzip"

        def inspect(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            if tree != REPO:
                facts["artifacts"][verified]["path"] = tree_rel(tree)
            return facts, error

        sandbox.inspect = inspect
        with sandbox:
            run_optimization(state)
        self.assertEqual(
            sandbox.contexts[0].paths, {verified: tree_rel(f"{ROOT}/base")}
        )
        self.assertIn("tune1", sandbox.jobs)
        for vid, jobs in sandbox.jobs.items():
            for job in jobs:
                with self.subTest(vid=vid, tree=job["cwd"]):
                    self.assertEqual(
                        job["cmd"].split(" ")[0],
                        f"{job['cwd']}/{tree_rel(job['cwd'])}",
                    )
                    self.assertNotIn("*", job["cmd"])
                    self.assertEqual(
                        job["input"], f"{ROOT}/inputs/atesor-measure.binary"
                    )

    def test_harness_build_gets_the_paths_of_its_tree(self) -> None:
        """A library in a hash directory: each install gets its path."""
        verified = "build/cachedObjs/114b1ec24f5c1d70f6731d54b5e28dc1/libz.a"
        state = _state()
        state.curated_artifacts[2]["filepath"] = f"{REPO}/{verified}"
        sandbox = _Sandbox(_profile(), workload=_harness_workload())
        original = sandbox.inspect

        def inspect(tree, artifacts, log, go, kind, read_log=False):
            facts, error = original(tree, artifacts, log, go, kind, read_log)
            if tree != REPO:
                facts["artifacts"][verified][
                    "path"
                ] = f"build/cachedObjs/{sandbox.vid(tree)}/libz.a"
            return facts, error

        sandbox.inspect = inspect
        with sandbox:
            run_optimization(state)
        self.assertEqual(
            sandbox.install_paths["tune1"],
            {verified: "build/cachedObjs/tune1/libz.a"},
        )
        self.assertEqual(
            sandbox.contexts[0].paths,
            {verified: "build/cachedObjs/base/libz.a"},
        )

    def test_package_tests_cannot_replace_the_tune_build(self) -> None:
        """An artifact that the tests rebuild gets the tune's build back."""
        sandbox = _Sandbox(_profile())
        calls = []

        def run_command(command, **kwargs):
            calls.append((command, kwargs.get("cwd")))
            if command[:3] == ["sh", "-c", optimize._RESTORE_SCRIPT]:
                return _ok("build/minigzip\n")
            return _ok()

        tested = []

        def run_tests(tree, env):
            tested.append(sandbox.vid(tree))
            return {"status": "passed"}

        lines = []
        with (
            sandbox,
            mock.patch.object(
                optimize, "execute_command", side_effect=run_command
            ),
        ):
            result = run_optimization(
                _state(), run_tests=run_tests, progress=lines.append
            )
        tune1 = _by_id(result)["tune1"]
        self.assertTrue(tune1["gate2"]["passed"])
        self.assertEqual(tune1["gate2"]["restored"], ["build/minigzip"])
        self.assertIn(
            "           the package tests rebuilt build/minigzip; "
            "Atesor put back the tune's build",
            lines,
        )
        self.assertIn("tune1", tested)
        artifacts = ["build/minigzip", "build/libz.so.1", "build/libz.a"]
        keep = ["sh", "-c", optimize._KEEP_SCRIPT, "sh"] + artifacts
        restore = ["sh", "-c", optimize._RESTORE_SCRIPT, "sh"] + artifacts
        self.assertIn((keep, f"{ROOT}/tune1"), calls)
        self.assertIn((restore, f"{ROOT}/tune1"), calls)
        # Each tree keeps its artifacts before its tests run.
        order = [
            command[2]
            for command, cwd in calls
            if cwd == f"{ROOT}/tune1" and command[:1] == ["sh"]
        ]
        self.assertEqual(
            order, [optimize._KEEP_SCRIPT, optimize._RESTORE_SCRIPT]
        )

    def test_artifacts_that_cannot_be_kept_fail_gate_2(self) -> None:
        """No copy aside, no package tests: the tree is not trusted."""
        sandbox = _Sandbox(_profile())

        def run_command(command, **kwargs):
            if command[:3] == ["sh", "-c", optimize._KEEP_SCRIPT]:
                return CommandResult(
                    "cmd", 1, "", "cp: No space left on device", 0.1
                )
            return _ok()

        tested = []
        with (
            sandbox,
            mock.patch.object(
                optimize, "execute_command", side_effect=run_command
            ),
        ):
            result = run_optimization(
                _state(),
                run_tests=lambda tree, env: tested.append(tree)
                or {"status": "passed"},
            )
        tune1 = _by_id(result)["tune1"]
        self.assertEqual(tune1["verdict"], "gate 2 failed")
        self.assertEqual(
            tune1["reason"],
            "the artifacts could not be copied aside before the package "
            "tests: cp: No space left on device",
        )
        self.assertEqual(tested, [])
        self.assertNotIn("tune1", [vid for vid, _ in sandbox.measured])

    def test_a_run_that_changes_its_input_gets_it_again(self) -> None:
        """The input file is written again; the next tunes are timed."""
        sandbox = _Sandbox(_profile())
        original = sandbox.measure

        def measure(base_job, variant_job, workload, pin, timeout):
            if sandbox.vid(variant_job["cwd"]) == "tune1":
                sandbox.measured.append(("tune1", pin))
                return {
                    "status": "failed",
                    "warnings": [],
                    "failed_job": "V",
                    "input_changed": True,
                    "reason": "the tune run changed its input file in "
                    "the warmup phase (the command must only "
                    "read {input})",
                }
            return original(base_job, variant_job, workload, pin, timeout)

        sandbox.measure = measure
        with (
            sandbox,
            mock.patch.object(
                bench, "write_input", return_value=None
            ) as writer,
        ):
            result = run_optimization(_state())
        verdicts = {vid: r["verdict"] for vid, r in _by_id(result).items()}
        self.assertEqual(verdicts["tune1"], "measurement failed")
        self.assertIn(
            "changed its input file", _by_id(result)["tune1"]["reason"]
        )
        writer.assert_any_call(
            "binary",
            8,
            "atesor-measure",
            f"{ROOT}/inputs/atesor-measure.binary",
        )
        self.assertEqual(verdicts["tune2"], "no change")

    def test_an_input_that_cannot_be_written_again_stops(self) -> None:
        """Without its input, no later tune can be timed."""
        sandbox = _Sandbox(_profile())

        def measure(base_job, variant_job, workload, pin, timeout):
            sandbox.measured.append((sandbox.vid(variant_job["cwd"]), pin))
            return {
                "status": "failed",
                "warnings": [],
                "failed_job": "V",
                "input_changed": True,
                "reason": "the tune run changed its input file",
            }

        sandbox.measure = measure
        with (
            sandbox,
            mock.patch.object(bench, "write_input", return_value="disk full"),
        ):
            result = run_optimization(_state())
        records = _by_id(result)
        self.assertEqual(records["tune1"]["verdict"], "measurement failed")
        for vid in ("tune2", "tune3", "tune4", "tune5"):
            self.assertEqual(records[vid]["verdict"], "skipped")
            self.assertEqual(
                records[vid]["reason"],
                "a run changed the input file and Atesor could not write "
                "it again: disk full",
            )
            self.assertNotIn(vid, sandbox.builds)
        self.assertNotIn("tune6", records, "no PGO without the input")
        self.assertEqual(result["status"], "not measured")

    def test_nothing_timed_is_not_a_lack_of_gain(self) -> None:
        """Tunes that fail to build give "not measured", not "no gain"."""
        sandbox = _Sandbox(
            _profile(),
            failing=(
                "tune1",
                "tune2",
                "tune3",
                "tune4",
                "tune5",
            ),
        )
        with sandbox:
            result = run_optimization(_state())
        self.assertEqual(result["status"], "not measured")
        self.assertIn(
            "every tune stopped before its timing: Tune-1: " "build failed",
            result["reason"],
        )

    def test_measure_respects_the_budget_and_qemu(self) -> None:
        """No session starts that the budget cannot finish."""
        record = {"variant": {"vid": "tune1"}, "tree": f"{ROOT}/tune1"}
        slow = bench.Workload(
            mode="tests",
            command="ctest",
            deterministic=False,
            run_seconds=40.0,
        )
        context = {
            "workload": slow,
            "kind": "native",
            "budget": optimize._Budget(600, lambda: 0.0),
            "state": _state(),
            "root": ROOT,
            "qemu_env": None,
        }
        with mock.patch.object(bench, "measure") as measure:
            result = optimize._measure(dict(record), {}, context)
        measure.assert_not_called()
        self.assertEqual(result["verdict"], "not measured")
        self.assertIn("about 37 min", result["reason"])
        context["kind"] = "qemu"
        with mock.patch.object(bench, "check_once") as check:
            result = optimize._measure(dict(record), {}, context)
        check.assert_not_called()
        self.assertEqual(result["verdict"], "built")

    def test_budget_below_one_step_builds_nothing(self) -> None:
        with _Sandbox(_profile()) as sandbox:
            result = run_optimization(
                _state(), budget_seconds=optimize.MIN_STEP_SECONDS - 1
            )
        self.assertEqual(result["status"], "not measured")
        self.assertIn("shorter than one step", result["reason"])
        self.assertEqual(sandbox.builds, {})

    def test_budget_skips_the_remaining_tunes(self) -> None:
        times = iter([0.0] + [0.0] * 6 + [10_000.0] * 200)
        with _Sandbox(_profile()) as sandbox:
            result = run_optimization(
                _state(), budget_seconds=3600, clock=lambda: next(times)
            )
        verdicts = [r["verdict"] for r in result["variants"]]
        self.assertIn("skipped", verdicts)
        self.assertIn("base", sandbox.builds)

    def _timed_sandbox(self, now, build_seconds, measure_seconds, **kwargs):
        """Return a sandbox whose builds and timings move a fake clock."""
        sandbox = _Sandbox(_profile(), **kwargs)
        build, measure = sandbox.build, sandbox.measure

        def slow_build(commands, repo_path, dest, env, timeout):
            now[0] += build_seconds(sandbox.vid(dest))
            return build(commands, repo_path, dest, env, timeout)

        def slow_measure(*args, **kw):
            now[0] += measure_seconds
            return measure(*args, **kw)

        sandbox.build, sandbox.measure = slow_build, slow_measure
        return sandbox

    def test_budget_keeps_time_for_the_second_check(self) -> None:
        """After a faster tune, later tunes wait; its check still runs.

        Base 600 s; Tune-1 900 s (faster) at 1500 s. Another tune (900 s)
        plus the check (900 s and a 600 s layout baseline) would end
        after 3600 s, so Tune-2 to Tune-6 skip and Tune-1 is accepted.
        """
        now = [0.0]
        sandbox = self._timed_sandbox(
            now,
            lambda vid: 600,
            300,
            ratios={
                "tune1": (0.8, "faster"),
                "tune1-layout2": (0.8, "faster"),
            },
        )
        with sandbox:
            result = run_optimization(
                _state(), budget_seconds=3600, clock=lambda: now[0]
            )
        records = _by_id(result)
        kept = (
            "the rest of the time budget is kept for the second check "
            "of Tune-1"
        )
        for vid in ("tune2", "tune3", "tune4", "tune5", "tune6"):
            with self.subTest(vid=vid):
                self.assertEqual(records[vid]["verdict"], "skipped")
                self.assertEqual(records[vid]["reason"], kept)
                self.assertNotIn(vid, sandbox.builds)
        self.assertEqual(records["tune1"]["seconds"], 900)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["accepted"], "tune1")
        self.assertIn("tune1-layout2", sandbox.builds)
        self.assertLessEqual(now[0], 3600)

    def test_gain_without_time_for_its_check_is_not_confirmed(self) -> None:
        """No time left for the second check: the gain stays open.

        It is no measured "no gain", so the cache does not keep it, and
        the fix line asks for a larger budget.
        """
        now = [0.0]
        sandbox = self._timed_sandbox(
            now, lambda vid: 1200, 1000, ratios={"tune1": (0.8, "faster")}
        )
        lines = []
        with sandbox:
            result = run_optimization(
                _state(),
                budget_seconds=3600,
                clock=lambda: now[0],
                progress=lines.append,
            )
        record = _by_id(result)["tune1"]
        self.assertEqual(record["verdict"], "not confirmed")
        self.assertEqual(record["unchecked"], "time")
        self.assertEqual(result["status"], "not confirmed")
        best = (
            "Tune-1, 20.0% faster (not confirmed: the time budget ended "
            "before the second check)"
        )
        self.assertEqual(
            result["headline"],
            f"NO: no faster tune was confirmed (best: {best})",
        )
        self.assertEqual(
            result["reason"],
            "faster in the first measurement, but the time budget ended "
            "before the second check: Tune-1",
        )
        self.assertEqual(
            result["advice"],
            "run again with a larger ATESOR_OPTIMIZE_BUDGET (this run had "
            "3600 s)",
        )
        self.assertIn(
            f"Result   : NO: no faster tune was confirmed (best: " f"{best})",
            lines,
        )
        self.assertIn(
            "           fix: run again with a larger "
            "ATESOR_OPTIMIZE_BUDGET (this run had 3600 s)",
            lines,
        )
        from src import memory

        self.assertFalse(
            memory.record_optimization(
                "zlib", result, machine={"core_name": "sifive-p550"}
            )
        )

    def test_second_check_without_time_is_not_a_failed_check(self) -> None:
        """A check that starts but cannot time does not say "did not hold"."""
        now = [0.0]
        sandbox = self._timed_sandbox(
            now,
            lambda vid: 1900 if vid == "base-layout2" else 600,
            300,
            ratios={"tune1": (0.8, "faster")},
        )
        with sandbox:
            result = run_optimization(
                _state(), budget_seconds=3600, clock=lambda: now[0]
            )
        record = _by_id(result)["tune1"]
        self.assertEqual(record["verdict"], "not confirmed")
        self.assertEqual(record["unchecked"], "time")
        self.assertTrue(
            record["reason"].startswith(
                "the second check in a second code layout"
            )
        )
        self.assertIn("gave no time: not measured", record["reason"])
        self.assertNotIn("tune1-layout2", [vid for vid, _ in sandbox.measured])
        self.assertEqual(result["status"], "not confirmed")
        self.assertNotIn("did not hold", result["headline"])

    def test_archive_only_library_compares_its_code(self) -> None:
        """A library of static archives only: its code counts too."""
        for code, expected in (
            (
                True,
                "its riscv64 artifact has the code of the verified " "build",
            ),
            (
                False,
                "it has no ELF file or static archive with code to " "compare",
            ),
        ):
            with self.subTest(code=code):
                state = _state()
                state.curated_artifacts = [
                    {
                        "filepath": f"{REPO}/build/libopus.a",
                        "type": "library_static",
                        "role": "primary",
                    }
                ]
                sandbox = _Sandbox(_profile())
                original = sandbox.inspect

                def inspect(
                    tree, artifacts, log, go, kind, read_log=False, code=code
                ):
                    facts, error = original(
                        tree, artifacts, log, go, kind, read_log
                    )
                    for info in facts["artifacts"].values():
                        info["archive"] = True
                        if code:
                            info["code_sha256"] = info["sha256"]
                    return facts, error

                sandbox.inspect = inspect
                lines = []
                with sandbox:
                    result = run_optimization(state, progress=lines.append)
                self.assertTrue(any(expected in line for line in lines), lines)
                self.assertFalse(any("all 0" in line for line in lines))
                self.assertEqual(
                    result["baseline"]["artifacts"], 1 if code else 0
                )

    def test_pgo_label_names_what_it_trains_on(self) -> None:
        """C/C++ trains on a second input; Go on its own benchmarks.

        A C/C++ workload without an input trains on the workload itself.
        """
        report = mock.Mock()
        context = {"root": ROOT, "report": report, "trees": []}
        cases = [
            ("go", "binary", "PGO, trained on the workload"),
            ("go", None, "PGO, trained on the workload"),
            ("c", "binary", "PGO, trained on a second input"),
            ("c", None, "PGO, trained on the workload"),
        ]
        for language, kind, value in cases:
            with self.subTest(language=language, kind=kind):
                workload = bench.Workload(
                    mode="gobench" if language == "go" else "harness",
                    command="{bin}",
                    input_kind=kind,
                    input_mb=8 if kind else 0,
                )
                base = {
                    "variant": Variant(
                        "tune1", "t", language, label="Tune-1"
                    ).to_dict()
                }
                record = optimize._pgo_variant(
                    base,
                    dict(context, workload=workload),
                    "Tune-2",
                    skip="kept",
                )
                self.assertEqual(record["verdict"], "skipped")
                self.assertEqual(record["variant"]["value"], value)
                report.line.assert_called_with(
                    "Tune-2",
                    f"profile-guided optimization = {value}, on "
                    "top of Tune-1",
                )

    def test_rust_build_with_its_own_rustflags(self) -> None:
        """A build that sets RUSTFLAGS keeps only the LTO tune."""
        state = _state(build_system="cargo")
        state.build_plan.phases[1].commands = [
            'RUSTFLAGS="-C opt-level=3" cargo build --release'
        ]
        with _Sandbox(_profile()):
            result = run_optimization(state)
        variants = [r["variant"] for r in result["variants"]]
        self.assertEqual(
            [(v["vid"], v["label"], v["technique"]) for v in variants],
            [("tune1", "Tune-1", "link-time optimization")],
        )
        self.assertNotIn("RUSTFLAGS", variants[0]["env"])
        self.assertTrue(any("RUSTFLAGS" in note for note in result["notes"]))


class TestKeepArtifacts(unittest.TestCase):
    """The scripts that keep a tree's artifacts, and their wrappers.

    The scripts run for real here; the wrapper test fakes execute_command.
    """

    def _sh(self, script, tree, rels):
        return subprocess.run(
            ["sh", "-c", script, "sh"] + rels,
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_scripts_put_back_what_the_tests_changed(self) -> None:
        """Rebuilt, re-linked and removed artifacts all come back."""
        with tempfile.TemporaryDirectory() as tree:

            def write(rel, text):
                os.makedirs(
                    os.path.dirname(os.path.join(tree, rel)), exist_ok=True
                )
                with open(os.path.join(tree, rel), "w") as fh:
                    fh.write(text)

            def read(rel):
                with open(os.path.join(tree, rel)) as fh:
                    return fh.read()

            write("programs/lz4", "tune build")
            os.chmod(os.path.join(tree, "programs/lz4"), 0o755)
            write("lib/liblz4.so.1.10.0", "tune library")
            os.symlink("liblz4.so.1.10.0", os.path.join(tree, "lib/liblz4.so"))
            write("lib/a b/liblz4.a", "tune archive")
            rels = ["programs/lz4", "lib/liblz4.so", "lib/a b/liblz4.a"]
            kept = self._sh(optimize._KEEP_SCRIPT, tree, rels)
            self.assertEqual(kept.returncode, 0, kept.stderr)
            # The package tests: a debug build in place, the link moved
            # to their own library, and the archive's directory removed.
            write("programs/lz4", "debug build")
            write("lib/tests/liblz4.so.1.10.0", "test library")
            os.remove(os.path.join(tree, "lib/liblz4.so"))
            os.symlink(
                "tests/liblz4.so.1.10.0", os.path.join(tree, "lib/liblz4.so")
            )
            os.remove(os.path.join(tree, "lib/a b/liblz4.a"))
            os.rmdir(os.path.join(tree, "lib/a b"))
            restored = self._sh(optimize._RESTORE_SCRIPT, tree, rels)
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(restored.stdout.splitlines(), rels)
            self.assertEqual(read("programs/lz4"), "tune build")
            self.assertTrue(
                os.access(os.path.join(tree, "programs/lz4"), os.X_OK)
            )
            self.assertEqual(read("lib/liblz4.so"), "tune library")
            self.assertFalse(
                os.path.islink(os.path.join(tree, "lib/liblz4.so"))
            )
            self.assertEqual(read("lib/a b/liblz4.a"), "tune archive")
            self.assertEqual(
                read("lib/tests/liblz4.so.1.10.0"), "test library"
            )
            again = self._sh(optimize._RESTORE_SCRIPT, tree, rels)
            self.assertEqual((again.returncode, again.stdout), (0, ""))

    def test_wrappers_report_errors(self) -> None:
        """A failed copy is an error; the restore lists what came back."""
        failed = CommandResult("cmd", 1, "", "cp: cannot stat 'x'", 0.1)
        with mock.patch.object(
            optimize, "execute_command", return_value=failed
        ):
            self.assertEqual(
                optimize._keep_artifacts("/t", ["x"]), "cp: cannot stat 'x'"
            )
            self.assertEqual(
                optimize._restore_artifacts("/t", ["x"]),
                ([], "cp: cannot stat 'x'"),
            )
        with mock.patch.object(optimize, "execute_command") as run:
            self.assertEqual(optimize._keep_artifacts("/t", []), "")
            self.assertEqual(optimize._restore_artifacts("/t", []), ([], ""))
        run.assert_not_called()


class TestResultContract(unittest.TestCase):
    """A real run_optimization result feeds the report and the zip."""

    def test_report_and_manifest_read_the_result(self) -> None:
        """Every tune reaches the report; the accepted one the zip."""
        import main
        from src.packager import package_build

        state = _state()
        sandbox = _Sandbox(
            _profile(),
            workload=_harness_workload(),
            ratios={
                "tune3": (0.90, "faster"),
                "tune3-layout2": (0.91, "faster"),
            },
        )
        with sandbox:
            result = run_optimization(state)
        report = main.generate_detailed_report(
            {
                "repo_name": "zlib",
                "build_status": "SUCCESS",
                "artifact_verification": {"status": "verified"},
                "optimization": result,
                "machine_profile": state.machine_profile,
            }
        )
        for record in result["variants"]:
            self.assertIn(f"| {record['variant']['label']} |", report)
        self.assertIn("**Result**: accepted: Tune-3", report)
        self.assertIn("core sifive-p550", report)
        self.assertIn("C harness by the optimizer agent", report)
        self.assertIn("### Workload source: `.atesor/bench/harness.c`", report)
        self.assertIn("| 0.9 (0.89-0.91) |", report)
        json.dumps(result)

        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repos", "zlib")
            accepted = os.path.join(tmp, "repos", "zlib.opt", "accepted")
            os.makedirs(os.path.join(accepted, "build"))
            os.makedirs(repo)
            for rel in result["accepted_artifacts"]:
                with open(os.path.join(accepted, rel), "wb") as handle:
                    handle.write(b"\x7fELF " + rel.encode())
            recipe = os.path.join(tmp, "recipe.md")
            with open(recipe, "w") as handle:
                handle.write("# recipe\n")
            path = package_build(
                repo_name="zlib",
                repo_path=repo,
                recipe_path=recipe,
                platform_name="debian-native",
                packages_dir=os.path.join(tmp, "packages"),
                optimization=result,
                machine=state.machine_profile,
                optimized_dir=accepted,
            )
            import zipfile

            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                names = archive.namelist()
        section = manifest["optimization"]
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(section["accepted"], "tune3")
        self.assertEqual(section["accepted_label"], "Tune-3")
        self.assertEqual(section["workload_mode"], "harness")
        self.assertEqual(section["accepted_variant"]["flags"][-1], "-O3")
        self.assertEqual(section["variants"][0]["technique"], "core tuning")
        self.assertEqual(manifest["machine"]["label"], "rv64gc+zba+zbb")
        self.assertEqual(len(section["artifacts"]), 3)
        self.assertIn("optimized/build/minigzip", names)
        self.assertIn("optimized/build/libz.a", names)


# Stands in for "docker exec" and "podman exec": it applies --env and -w,
# drops the container name, and runs the command on this host.
_FAKE_ENGINE = """#!{python}
import os
import sys

args = sys.argv[1:]
assert args.pop(0) == "exec", sys.argv
env = dict(os.environ)
cwd = None
while args and args[0].startswith("-"):
    flag = args.pop(0)
    if flag == "--env":
        name, equals, value = args.pop(0).partition("=")
        if equals:
            env[name] = value
    elif flag == "-w":
        cwd = args.pop(0)
args.pop(0)
if cwd:
    os.chdir(cwd)
os.execvpe(args[0], args, env)
"""


class TestSandboxPath(unittest.TestCase):
    """The programs survive the real execute_command quoting.

    QEMU runs "docker exec ... bash -c '<timeout bash -c ...>'". Native
    sends "exec podman exec ..." to the machine's shell over ssh. Fake
    engines run both paths on this host; ssh becomes "sh -c".
    """

    def setUp(self) -> None:
        """Put fake docker and podman first on PATH."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        bin_dir = os.path.join(self.dir, "bin")
        os.makedirs(bin_dir)
        for name in ("docker", "podman"):
            path = os.path.join(bin_dir, name)
            with open(path, "w") as handle:
                handle.write(_FAKE_ENGINE.format(python=sys.executable))
            os.chmod(path, 0o755)
        path_env = mock.patch.dict(
            os.environ,
            {"PATH": bin_dir + os.pathsep + os.environ.get("PATH", "")},
        )
        path_env.start()
        self.addCleanup(path_env.stop)

    def _patches(self, native: bool):
        from src import target, tools

        patches = [
            mock.patch.object(tools, "is_native", return_value=native),
            mock.patch.object(target, "is_native", return_value=native),
        ]
        if native:
            patches += [
                mock.patch.object(target, "ssh_argv", return_value=["sh"]),
                mock.patch.object(target, "ssh_alias", return_value="-c"),
            ]
        else:
            patches.append(
                mock.patch.object(
                    tools.DockerConfig,
                    "is_container_running",
                    return_value=True,
                )
            )
        return patches

    def test_programs_on_both_targets(self) -> None:
        """Prepare, inspect, finish, the runner and the input writer."""
        for native in (False, True):
            with self.subTest(native=native), contextlib.ExitStack() as st:
                for patch in self._patches(native):
                    st.enter_context(patch)
                self._exercise(os.path.join(self.dir, f"t{int(native)}"))

    def _exercise(self, base: str) -> None:
        src = os.path.join(base, "repo")
        os.makedirs(os.path.join(src, "bin"))
        with open(os.path.join(src, "main.c"), "w") as handle:
            handle.write("int main(void) { return 0; }\n")
        root = src + ".opt"
        dest = os.path.join(root, "v1-tune")
        self.assertIsNone(optimize.prepare_tree(src, dest, root))
        self.assertTrue(
            os.access(os.path.join(dest, ".atesor", "cc"), os.X_OK)
        )
        image = _elf(_code("sh3add a0, a1, a2", "c.zext.b a0"))
        with open(os.path.join(dest, "bin", "tool"), "wb") as handle:
            handle.write(image)
        facts, error = optimize.inspect_tree(
            dest,
            ["bin/tool"],
            os.path.join(dest, ".atesor", "cc.log"),
            False,
            "native",
        )
        self.assertEqual(error, "")
        self.assertEqual(facts["cc_calls"], 0)
        self.assertEqual(
            facts["artifacts"]["bin/tool"]["counts"], {"zba": 1, "zcb": 1}
        )
        out = os.path.join(base, "out.txt")
        value = 'a\'b "c" $d `e`'
        payload = bench.run_jobs(
            {
                "jobs": {
                    "B": {
                        "cmd": 'printf "%s|%s" "$X" "$QEMU_CPU" > ' + out,
                        "env": {"X": value},
                    }
                },
                "rounds": 1,
                "seed": 1,
                "hash_output": True,
            },
            timeout=120,
            extra_env={"QEMU_CPU": "rv64i,m=true,zba=true"},
        )
        self.assertIsNone(payload["error"], payload)
        with open(out) as handle:
            self.assertEqual(handle.read(), value + "|rv64i,m=true,zba=true")
        sample = os.path.join(base, "input.txt")
        self.assertIsNone(bench.write_input("text", 1, "seed A", sample))
        self.assertGreaterEqual(os.path.getsize(sample), 1 << 20)
        source = 'printf("ATESOR-CHECK %d\\n", 1); // it\'s "quoted" $x\n'
        self.assertIsNone(
            workload_module.write_files(
                dest, {".atesor/bench/harness.c": source}
            )
        )
        with open(os.path.join(dest, ".atesor", "bench", "harness.c")) as fh:
            self.assertEqual(fh.read(), source)
        self.assertIn(
            "refusing to write outside the tree",
            workload_module.write_files(dest, {"../x.c": "x"}),
        )
        copied, error = optimize._run_program(
            optimize._FINISH_PROGRAM,
            {"root": root, "reset": True},
            timeout=120,
        )
        self.assertEqual(copied, {"copied": []})
        self.assertFalse(os.path.exists(root))


if __name__ == "__main__":
    unittest.main()
