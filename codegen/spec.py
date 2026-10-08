"""Model of the pinned webgpu.json plus the hand tables in abi.json.

Naming and layout rules mirror the upstream Go generator (webgpu-headers/gen) so
that every name produced here is the exact C identifier in webgpu.h. The
generated guest/tests/abi_check.gen.c makes the compiler prove that.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# wasm32 C ABI: (C type, size, align, core wasm type)
PRIMS = {
    "bool": ("WGPUBool", 4, 4, "i32"),
    "uint16": ("uint16_t", 2, 2, "i32"),
    "uint32": ("uint32_t", 4, 4, "i32"),
    "uint64": ("uint64_t", 8, 8, "i64"),
    "usize": ("size_t", 4, 4, "i32"),
    "int16": ("int16_t", 2, 2, "i32"),
    "int32": ("int32_t", 4, 4, "i32"),
    "float32": ("float", 4, 4, "f32"),
    "nullable_float32": ("float", 4, 4, "f32"),
    "float64": ("double", 8, 8, "f64"),
    "float64_supertype": ("double", 8, 8, "f64"),
}
STRINGS = ("nullable_string", "string_with_default_empty", "out_string")
PTR = 4  # sizeof(void*) == sizeof(size_t) on wasm32

RUST_KEYWORDS = {"type", "loop", "match", "ref", "in", "mod", "move", "use", "fn", "impl", "self", "override", "final"}


def pascal(s):
    out, upper = [], True
    for c in s:
        if upper:
            out.append(c.upper())
            upper = False
        elif c == "_":
            upper = True
        else:
            out.append(c)
    return "".join(out)


def camel(s):
    out, upper = [], False
    for c in s:
        if upper:
            out.append(c.upper())
            upper = False
        elif c == "_":
            upper = True
        else:
            out.append(c)
    return "".join(out)


def singular(s):
    if s == "entries":
        return "entry"
    return s[:-1] if s.endswith("s") else s


def snake(s):
    """Rust identifier for a spec name (spec names are snake_case with acronyms)."""
    s = s.lower()
    return s + "_" if s in RUST_KEYWORDS else s


def rust_variant(entry):
    p = pascal(entry)
    if p[0].isdigit():
        m = re.match(r"^(\d)D(.*)$", p)
        return "D" + m.group(1) + m.group(2) if m else "N" + p
    return p


def is_void(base):
    return base == "c_void" or base.startswith("c_void_")


def parse_int(v):
    return int(v, 16) if isinstance(v, str) and v.startswith("0x") else int(v)


class Ty(object):
    """A member, argument or return type reference."""

    def __init__(self, d):
        raw = d["type"]
        m = re.match(r"^array<(.*)>$", raw)
        self.raw = raw
        self.array = bool(m)
        base = m.group(1) if m else raw
        self.pointer = d.get("pointer")
        self.optional = bool(d.get("optional"))
        self.name = d.get("name")
        self.doc = d.get("doc", "")
        self.default = d.get("default")
        if base in PRIMS:
            self.kind, self.base = "prim", base
        elif base in STRINGS:
            self.kind, self.base = "string", base
        elif is_void(base):
            self.kind, self.base = "void", base
        else:
            self.kind, self.base = base.split(".", 1)

    def c_base(self):
        if self.kind == "prim":
            return PRIMS[self.base][0]
        if self.kind == "string":
            return "WGPUStringView"
        if self.kind == "void":
            return "void"
        if self.kind == "callback":
            return "WGPU" + pascal(self.base) + "CallbackInfo"
        return "WGPU" + pascal(self.base)

    def c_type(self):
        t = self.c_base()
        if self.pointer == "immutable":
            return t + " const *"
        if self.pointer == "mutable":
            return t + " *"
        return t


class Member(object):
    def __init__(self, d):
        self.ty = Ty(d)
        self.name = d["name"]
        self.offset = None  # of the first field (the count, for arrays)

    @property
    def c_name(self):
        return camel(self.name)

    @property
    def c_count_name(self):
        return camel(singular(self.name)) + "Count"


class Struct(object):
    def __init__(self, d, extension=False):
        self.name = d["name"]
        self.kind = d["type"]  # extensible | extensible_callback_arg | extension | standalone
        self.extends = list(d.get("extends") or [])
        self.free_members = bool(d.get("free_members"))
        self.doc = d.get("doc", "")
        self.members = [Member(m) for m in d.get("members", [])]
        self.is_extension_api = extension  # declared by webgpu_wasix.h, not webgpu.h
        self.size = self.align = None
        self.stype = None

    @property
    def c_name(self):
        return "WGPU" + pascal(self.name)

    @property
    def rust_name(self):
        return pascal(self.name)

    @property
    def has_chain_root(self):
        return self.kind in ("extensible", "extensible_callback_arg")


class Enum(object):
    def __init__(self, d):
        self.name = d["name"]
        self.entries = []  # (name, value)
        for i, e in enumerate(d["entries"]):
            if e is None:
                continue
            value = parse_int(e["value"]) if "value" in e else i
            if e.get("namespace") == "compatibility_mode":
                value |= 0x2000 << 16
            self.entries.append((e["name"], value))

    @property
    def c_name(self):
        return "WGPU" + pascal(self.name)

    @property
    def rust_name(self):
        return pascal(self.name)

    def c_entry(self, entry):
        return self.c_name + "_" + pascal(entry)


class Bitflag(object):
    def __init__(self, d):
        self.name = d["name"]
        raw = d["entries"]
        values = {}
        for i, e in enumerate(raw):
            if "value_combination" in e:
                continue
            values[e["name"]] = parse_int(e["value"]) if "value" in e else (0 if i == 0 else 1 << (i - 1))
        self.entries = []
        for e in raw:
            if "value_combination" in e:
                v = 0
                for n in e["value_combination"]:
                    v |= values[n]
            else:
                v = values[e["name"]]
            self.entries.append((e["name"], v))

    @property
    def c_name(self):
        return "WGPU" + pascal(self.name)


class Callback(object):
    def __init__(self, d):
        self.name = d["name"]
        self.style = d["style"]  # callback_mode | immediate
        self.args = [Ty(a) for a in d.get("args", [])]

    @property
    def info_size(self):
        return 20 if self.style == "callback_mode" else 16


class Fn(object):
    """One C function of the API, flattened the way the header declares it."""

    def __init__(self, c_name, owner, name, args, returns, callback, role, extension=False):
        self.c_name = c_name
        self.owner = owner  # spec name of the object (or struct for free_members), None for globals
        self.name = name
        self.args = args  # [Ty], excluding the receiver
        self.returns = returns  # Ty or None
        self.callback = callback  # spec callback name or None
        self.role = role  # global | method | add_ref | release | free_members | get_proc_address
        self.is_extension_api = extension

    @property
    def rust_name(self):
        if self.owner is None:
            return self.name.lower()
        return (self.owner + "_" + self.name).lower()

    def c_return(self):
        if self.callback:
            return "WGPUFuture"
        if self.role == "get_proc_address":
            return "WGPUProc"
        if self.returns is None:
            return "void"
        return self.returns.c_type()

    def c_params(self):
        """[(c type, c name)] exactly as declared, arrays expanded to count + pointer."""
        out = []
        if self.role == "free_members":
            out.append(("WGPU" + pascal(self.owner), camel(self.owner)))
            return out
        if self.owner is not None:
            out.append(("WGPU" + pascal(self.owner), camel(self.owner)))
        for a in self.args:
            if a.array:
                out.append(("size_t", camel(singular(a.name)) + "Count"))
            out.append((a.c_type(), camel(a.name)))
        if self.callback:
            out.append(("WGPU" + pascal(self.callback) + "CallbackInfo", "callbackInfo"))
        return out

    def c_decl(self, name=None):
        params = ", ".join("%s %s" % p for p in self.c_params()) or "void"
        return "%s %s(%s)" % (self.c_return(), name or self.c_name, params)

    def c_call_args(self):
        return ", ".join(n for _, n in self.c_params())


class Spec(object):
    def __init__(self):
        with open(os.path.join(ROOT, "spec", "webgpu.json")) as f:
            self.raw = json.load(f)
        with open(os.path.join(ROOT, "spec", "abi.json")) as f:
            self.abi = json.load(f)
        ext = self.abi["extension"]

        self.enums = [Enum(e) for e in self.raw["enums"]]
        self.enum_by_name = dict((e.name, e) for e in self.enums)
        self.bitflags = [Bitflag(b) for b in self.raw["bitflags"]]
        self.constants = [(c["name"], c["value"]) for c in self.raw["constants"]]
        self.callbacks = [Callback(c) for c in self.raw["callbacks"]]
        self.callback_by_name = dict((c.name, c) for c in self.callbacks)
        self.structs = [Struct(s) for s in self.raw["structs"]]
        self.structs += [Struct(s, extension=True) for s in ext["structs"]]
        self.struct_by_name = dict((s.name, s) for s in self.structs)
        self.objects = [o["name"] for o in self.raw["objects"]]
        self.lenient_enums = set(self.abi["lenient_enums"])

        for name, extra in self.abi.get("extra_extends", {}).items():
            s = self.struct_by_name[name]
            s.extends = sorted(set(s.extends) | set(extra))

        # sType values: positional in enum s_type, plus the extension block.
        stypes = dict(self.enum_by_name["s_type"].entries)
        for name, value in ext["stypes"].items():
            stypes[name] = parse_int(value)
        self.extension_stypes = dict((n, parse_int(v)) for n, v in ext["stypes"].items())
        self.stype_aliases = dict((parse_int(k), v) for k, v in ext.get("stype_aliases", {}).items())
        for s in self.structs:
            if s.kind == "extension":
                s.stype = stypes[s.name]

        for s in self.structs:
            self._layout(s)

        self.functions = self._flatten_functions(ext)
        self.fn_by_c_name = dict((f.c_name, f) for f in self.functions)

        self.shim_only = set(self.abi["shim_only"])
        self.shim_with_import = set(self.abi["shim_with_import"])
        for n in self.shim_only | self.shim_with_import:
            if n not in self.fn_by_c_name:
                raise SystemExit("abi.json names unknown function " + n)
        self.special_imports = self.abi["special_imports"]
        self.suspending = set(self.abi.get("suspending_imports", []))
        self.suspending |= set(s["name"] for s in self.special_imports if s.get("suspending"))

    # -- layout -------------------------------------------------------------

    def size_align(self, ty):
        """Size and alignment of one non-array member of this type."""
        if ty.pointer:
            return PTR, PTR
        if ty.kind == "prim":
            return PRIMS[ty.base][1], PRIMS[ty.base][2]
        if ty.kind == "string":
            return 8, 4
        if ty.kind in ("enum", "object"):
            return 4, 4
        if ty.kind == "bitflag":
            return 8, 8
        if ty.kind == "void":
            return PTR, PTR
        if ty.kind == "callback":
            return self.callback_by_name[ty.base].info_size, 4
        if ty.kind == "struct":
            s = self.struct_by_name[ty.base]
            if s.size is None:
                self._layout(s)
            return s.size, s.align
        raise SystemExit("no layout for " + ty.raw)

    def _layout(self, s):
        if s.size is not None:
            return
        off, align = 0, 1
        if s.has_chain_root:
            off, align = PTR, PTR
        elif s.kind == "extension":
            off, align = 8, 4
        for m in s.members:
            if m.ty.array:
                sz, al = 2 * PTR, PTR
            else:
                sz, al = self.size_align(m.ty)
            off = (off + al - 1) // al * al
            m.offset = off
            off += sz
            align = max(align, al)
        s.align = align
        s.size = (off + align - 1) // align * align

    # -- functions ----------------------------------------------------------

    def _flatten_functions(self, ext):
        fns = []

        def make(owner, d, role, extension=False):
            cb = d.get("callback")
            cb = cb.split(".", 1)[1] if cb else None
            ret = Ty(d["returns"]) if d.get("returns") else None
            args = [Ty(a) for a in d.get("args", [])]
            if owner is None:
                c_name = "wgpu" + pascal(d["name"])
            else:
                c_name = "wgpu" + pascal(owner) + pascal(d["name"])
            if "c_name" in d:
                c_name = d["c_name"]
            return Fn(c_name, owner, d["name"], args, ret, cb, role, extension)

        for f in self.raw["functions"]:
            fns.append(make(None, f, "global"))
        fns.append(Fn("wgpuGetProcAddress", None, "get_proc_address",
                      [Ty({"name": "proc_name", "type": "string_with_default_empty"})], None, None,
                      "get_proc_address"))
        for o in self.raw["objects"]:
            for m in o.get("methods", []):
                fns.append(make(o["name"], m, "method"))
            fns.append(Fn("wgpu" + pascal(o["name"]) + "AddRef", o["name"], "add_ref", [], None, None, "add_ref"))
            fns.append(Fn("wgpu" + pascal(o["name"]) + "Release", o["name"], "release", [], None, None, "release"))
        for s in self.structs:
            if s.free_members:
                fns.append(Fn("wgpu" + pascal(s.name) + "FreeMembers", s.name, "free_members", [], None, None,
                              "free_members"))
        for f in ext.get("functions", []):
            fns.append(make(f["object"], f, "method", extension=True))
        return fns

    def raw_import_functions(self):
        """Functions the host implements under their own C name."""
        return [f for f in self.functions if f.c_name not in self.shim_only]

    def forwarded_functions(self):
        """Functions whose guest definition is a generated one-line forwarder."""
        return [f for f in self.functions
                if f.c_name not in self.shim_only and f.c_name not in self.shim_with_import]

    # -- wasm lowering ------------------------------------------------------

    def lower_params(self, fn):
        """[(name, core wasm type)] of fn under the wasm32 C ABI."""
        out = []
        if fn.role == "free_members":
            return [(snake(fn.owner), "i32")]
        if fn.owner is not None:
            out.append((snake(fn.owner), "i32"))
        for a in fn.args:
            n = snake(a.name)
            if a.array:
                out.append((snake(singular(a.name)) + "_count", "i32"))
                out.append((n, "i32"))
            elif a.pointer or a.kind in ("string", "struct", "object", "enum", "void", "callback"):
                # by-value aggregates are passed indirectly
                out.append((n, "i32"))
            elif a.kind == "bitflag":
                out.append((n, "i64"))
            else:
                out.append((n, PRIMS[a.base][3]))
        if fn.callback:
            out.append(("callback_info", "i32"))
        return out

    def lower_result(self, fn):
        if fn.callback:
            return "i64"  # WGPUFuture is a single-u64 struct
        r = fn.returns
        if fn.role == "get_proc_address":
            return "i32"
        if r is None:
            return None
        if r.pointer or r.kind in ("object", "enum", "void"):
            return "i32"
        if r.kind == "bitflag":
            return "i64"
        if r.kind == "prim":
            return PRIMS[r.base][3]
        if r.kind == "struct" and r.base == "future":
            return "i64"
        raise SystemExit("cannot lower result of " + fn.c_name)
