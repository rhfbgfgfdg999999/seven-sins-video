#!/usr/bin/env bash
# usage: ./render_all.sh   -> out/NN_name.mp4（音效由程序按节拍网格合成并混入）
set -e
cd "$(dirname "$0")"
export SINS_WORK="${SINS_WORK:-$PWD/work}"; mkdir -p "$SINS_WORK" out
PY=$(command -v python3 || command -v python)
one(){ k=$1; n=$2
  "$PY" make.py $k --out "$SINS_WORK/$n.mp4"
  cp "$SINS_WORK/$n.mp4" out/$n.mp4
}
(one pride 01_傲慢; one wrath 03_暴怒; one greed 05_贪婪; one lust 07_色欲) &
(one envy 02_嫉妒; one sloth 04_懒惰; one gluttony 06_暴食) &
wait
