#!/usr/bin/env bash
# z420_ram_download.sh — RAM-staged download for the Z420: fetch into /dev/shm (tmpfs, RAM), then write to the 8 TB HDD
# in one sequential pass, verify sha256, and rename into place. The HDD only ever sees large sequential writes; the
# NVMe is never touched.
#   z420_ram_download.sh URL DEST [SHA256]
# If DEST.part exists on the HDD (an earlier, interrupted download), only the bytes after it are fetched (HTTP Range)
# into RAM and appended. Needs the remaining size free in /dev/shm.
set -euo pipefail
url=$1; dest=$2; want=${3:-}
shm=/dev/shm/z420_dl_$(basename "$dest").part
part=$dest.part
have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
total=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
[ -n "$total" ] || { echo "no Content-Length for $url" >&2; exit 1; }
need=$((total - have))
free=$(df --output=avail -B1 /dev/shm | tail -1)
echo "total $total B, on HDD $have B, to fetch $need B into RAM ($free B free in /dev/shm)"
[ "$need" -le "$free" ] || { echo "not enough room in /dev/shm" >&2; exit 1; }
if [ "$need" -gt 0 ]; then
    # resumable inside RAM too: -C - continues the shm file if this script is re-run
    shm_have=0; [ -f "$shm" ] && shm_have=$(stat -c %s "$shm")
    curl -fL --retry 20 --retry-delay 5 --retry-all-errors -r "$((have + shm_have))-" -o - "$url" >> "$shm"
    got=$(stat -c %s "$shm")
    [ "$got" -eq "$need" ] || { echo "RAM part is $got B, expected $need B (re-run to resume)" >&2; exit 1; }
    echo "in RAM: $got B; writing to HDD sequentially"
    dd if="$shm" of="$part" bs=64M oflag=append conv=notrunc status=progress
    sync "$part"
    rm -f "$shm"
fi
[ "$(stat -c %s "$part")" -eq "$total" ] || { echo "size mismatch on HDD" >&2; exit 1; }
if [ -n "$want" ]; then
    got=$(sha256sum "$part" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { echo "sha256 mismatch: $got != $want" >&2; exit 1; }
    echo "sha256 ok"
fi
mv "$part" "$dest"
echo "done: $dest"
