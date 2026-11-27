#!/usr/bin/env bash
# Print current whole-device VRAM used in bytes via sysfs (instant, falls back to rocm-smi).
for f in /sys/class/drm/card*/device/mem_info_vram_used; do
  if [[ -r "$f" ]]; then
    cat "$f"
    exit 0
  fi
done
rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
for _,v in d.items():
    print(int(v.get("VRAM Total Used Memory (B)", 0)))
    break
' 2>/dev/null || echo 0
