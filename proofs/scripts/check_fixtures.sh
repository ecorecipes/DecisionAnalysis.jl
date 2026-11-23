#!/bin/sh
# Run `check_certificate` on every fixture; the expected exit code is fixed by the file name:
# `reject_*` must exit 1, `malformed_*` must exit 2, everything else must exit 0.
# Run from proofs/ after `lake build check_certificate`.
status=0
for f in certificate-fixtures/*.json; do
  name=$(basename "$f")
  case "$name" in
    reject_*) want=1 ;;
    malformed_*) want=2 ;;
    *) want=0 ;;
  esac
  .lake/build/bin/check_certificate "$f" > /dev/null 2>&1
  got=$?
  if [ "$got" -eq "$want" ]; then
    echo "ok    $name (exit $got)"
  else
    echo "FAIL  $name (exit $got, expected $want)"
    status=1
  fi
done
exit $status
