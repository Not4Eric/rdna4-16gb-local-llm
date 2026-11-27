#!/usr/bin/env python3
"""Transplant the 15 self-distilled blk.64.* MTP tensors from a source GGUF into a copy of a
target Qwen 3.8 27B GGUF in-place (keeping all blk.0..63 weights and the full 248,320 vocabulary
bit-for-bit identical). Standard library only.

Usage:
  python3 scripts/transplant-mtp1.py --base BASE.gguf --donor DONOR.gguf --out OUT.gguf
"""
import argparse, hashlib, os, shutil, struct

def parse_gguf_tensors(path):
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        assert magic == b"GGUF", f"Not a GGUF file: {path}"
        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")
        def rval(vt):
            if vt == 0: return struct.unpack("<B", f.read(1))[0]
            if vt == 1: return struct.unpack("<b", f.read(1))[0]
            if vt == 2: return struct.unpack("<H", f.read(2))[0]
            if vt == 3: return struct.unpack("<h", f.read(2))[0]
            if vt == 4: return struct.unpack("<I", f.read(4))[0]
            if vt == 5: return struct.unpack("<i", f.read(4))[0]
            if vt == 6: return struct.unpack("<f", f.read(4))[0]
            if vt == 7: return struct.unpack("<?", f.read(1))[0]
            if vt == 8: return rstr()
            if vt == 9:
                at, al = struct.unpack("<IQ", f.read(12))
                for _ in range(al): rval(at)
                return None
            if vt == 10: return struct.unpack("<Q", f.read(8))[0]
            if vt == 11: return struct.unpack("<q", f.read(8))[0]
            if vt == 12: return struct.unpack("<d", f.read(8))[0]
            raise ValueError(f"Unknown GGUF value type {vt}")
        align = 32
        for _ in range(n_kv):
            k = rstr()
            vt = struct.unpack("<I", f.read(4))[0]
            v = rval(vt)
            if k == "general.alignment" and isinstance(v, int):
                align = v
        raw = []
        for _ in range(n_tensors):
            name = rstr()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(nd))
            ttype, offset = struct.unpack("<IQ", f.read(12))
            raw.append((name, dims, ttype, offset))
        data_start = (f.tell() + align - 1) & ~(align - 1)
    file_sz = os.path.getsize(path)
    by_off = sorted(raw, key=lambda x: x[3])
    out = {}
    for i, (name, dims, ttype, off) in enumerate(by_off):
        nxt = by_off[i + 1][3] if i + 1 < len(by_off) else (file_sz - data_start)
        out[name] = (dims, ttype, data_start + off, nxt - off)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Target full-vocab Qwen 3.8 27B GGUF (left unmodified)")
    ap.add_argument("--donor", required=True, help="Donor GGUF containing self-distilled blk.64.* MTP1 tensors")
    ap.add_argument("--out", required=True, help="Output GGUF path")
    ap.add_argument("--prefix", default="blk.64.", help="Tensor name prefix to transplant (default: blk.64.)")
    args = ap.parse_args()

    base_t = parse_gguf_tensors(args.base)
    donor_t = parse_gguf_tensors(args.donor)
    targets = sorted(k for k in base_t if k.startswith(args.prefix))
    assert targets, f"No tensors with prefix {args.prefix!r} found in {args.base}"

    for name in targets:
        assert name in donor_t, f"Missing {name} in donor GGUF"
        bd, bt, _, bsz = base_t[name]
        dd, dt, _, dsz = donor_t[name]
        assert (bd, bt, bsz) == (dd, dt, dsz), f"Mismatch for {name}: base={(bd, bt, bsz)} vs donor={(dd, dt, dsz)}"

    print(f"Copying {args.base} -> {args.out} ...")
    shutil.copyfile(args.base, args.out)

    with open(args.donor, "rb") as fd, open(args.out, "r+b") as fo:
        for name in targets:
            _, _, doff, sz = donor_t[name]
            _, _, boff, _ = base_t[name]
            fd.seek(doff)
            buf = fd.read(sz)
            fo.seek(boff)
            fo.write(buf)
            fo.flush()
            fo.seek(boff)
            verify = fo.read(sz)
            assert hashlib.sha256(buf).digest() == hashlib.sha256(verify).digest(), f"SHA-256 mismatch on {name}"
            print(f"  transplanted {name:36s} ({sz/2**20:6.2f} MiB) sha256={hashlib.sha256(buf).hexdigest()[:12]}")

    print(f"Done: transplanted and verified {len(targets)}/{len(targets)} tensors into {args.out}")

if __name__ == "__main__":
    main()
