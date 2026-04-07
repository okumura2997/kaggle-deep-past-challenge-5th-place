#!/usr/bin/env bash
# ======================================================================
# Unified AKT PDF Extraction Pipeline
# ======================================================================
# extract_unified.py を使って全PDFから翻字・翻訳ペアを抽出する。
# 各PDFのパスとプロファイルをハードコードして順次実行。
# 1つのPDFが失敗しても他は続行する。
#
# ── Pipeline (per PDF) ────────────────────────────────────────────────
#   Phase 1: detect   — ページごとに発掘番号/見出しを検出
#   Phase 2: link     — published_texts.csv と照合して transliteration_orig 取得
#   Phase 3: extract  — 翻字 + 翻訳を抽出 (multi-sample, ChrF++ consensus)
#   Phase 4: translate — 英語に翻訳 (英語出版のボリュームではスキップ)
#
# ── Output ────────────────────────────────────────────────────────────
#   data/extract_unified/{pdf_stem}/
#     profile.yaml, detections.csv, linked.csv,
#     pairs_raw.csv, pairs_en.csv, pairs_final.csv
#
# ── Usage ─────────────────────────────────────────────────────────────
#   bash scripts/run_extraction_pipeline.sh              # 全PDF実行
#   bash scripts/run_extraction_pipeline.sh --dry-run    # リスト表示のみ
# ======================================================================

set -Eeuo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

# ── Options ──────────────────────────────────────────────────────────
MODEL="openrouter/qwen/qwen3.5-plus-02-15"
TRANSLATE_MODEL="openrouter/google/gemini-3.1-flash-lite-preview"
MAX_CONCURRENCY=32
OUTPUT_ROOT="./data/extract_unified"
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)            MODEL="$2"; shift 2 ;;
    --translate-model)  TRANSLATE_MODEL="$2"; shift 2 ;;
    --max-concurrency)  MAX_CONCURRENCY="$2"; shift 2 ;;
    --output-root)      OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=true; shift ;;
    --skip-detect)      EXTRA_ARGS+=(--skip-detect); shift ;;
    --skip-extract)     EXTRA_ARGS+=(--skip-extract); shift ;;
    --skip-translate)   EXTRA_ARGS+=(--skip-translate); shift ;;
    --retry-empty)      EXTRA_ARGS+=(--retry-empty); shift ;;
    --max-pages)        EXTRA_ARGS+=(--max-pages "$2"); shift 2 ;;
    *)                  echo "[error] Unknown argument: $1"; exit 1 ;;
  esac
done

# ── PDF list ─────────────────────────────────────────────────────────
# Format: "pdf_path|profile_override"
# profile_override が空の場合はファイル名から自動解決
PDFS=(
  # AKT 1 (Turkish, _fixed版を使用)
  "input/pdfs/AKT_1_1990_fixed.pdf|"
  # AKT 2 (Turkish, _fixed版を使用)
  "input/pdfs/AKT_2_1995_fixed.pdf|"
  # AKT 3 (German, _fixed版を使用)
  "input/pdfs/AKT_3_1995_fixed.pdf|"
  # AKT 4 (Turkish)
  "input/pdfs/AKT 4 2006.pdf|"
  # AKT 5 (English, side-by-side)
  "input/pdfs/AKT 5 2008.pdf|"
  # AKT 6A–6E (English, side-by-side)
  "input/pdfs/AKT 6a.pdf|"
  "input/pdfs/AKT 6b.pdf|"
  "input/pdfs/AKT 6c.pdf|"
  "input/pdfs/AKT 6d.pdf|"
  "input/pdfs/AKT 6e.pdf|"
  # AKT 7A, 7B (Turkish)
  "input/pdfs/AKT 7a.pdf|"
  "input/pdfs/AKT 7b.pdf|"
  # AKT 8 (English, side-by-side)
  "input/pdfs/AKT 8 2015.pdf|"
  # AKT 9A (Turkish)
  "input/pdfs/AKT 9a.pdf|"
  # AKT 10 (Turkish)
  "input/pdfs/AKT 10.pdf|"
  # AKT 11A, 11B (Turkish)
  "input/pdfs/AKT 11a.pdf|"
  "input/pdfs/AKT 11b.pdf|"
  # AKT 12 (English, side-by-side)
  "input/pdfs/AKT_12.pdf|"
  # Larsen 2002 (English, side-by-side, Larsen heading detection)
  "input/pdfs/Larsen 2002 - The Assur-nada Archive. PIHANS 96 2002.pdf|Larsen 2002"
  # ICK 4 (German, requires explicit profile)
  "input/pdfs/Hecker Kryszat Matous - Kappadokische Keilschrifttafeln aus den Sammlungen der Karlsuniversitat Prag. ICK 4 1998.pdf|ick4"
)

# ── Dry-run: list PDFs and exit ──────────────────────────────────────
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run — PDFs to process:"
  for entry in "${PDFS[@]}"; do
    IFS='|' read -r pdf_path profile_override <<< "${entry}"
    if [[ -f "${pdf_path}" ]]; then
      status="OK"
    else
      status="MISSING"
    fi
    if [[ -n "${profile_override}" ]]; then
      echo "  [${status}] ${pdf_path}  (profile: ${profile_override})"
    else
      echo "  [${status}] ${pdf_path}"
    fi
  done
  exit 0
fi

# ── Run pipeline ─────────────────────────────────────────────────────
FAILED=()
SUCCEEDED=()
SKIPPED=()

echo "========================================"
echo " Unified Extraction Pipeline"
echo " Model:           ${MODEL}"
echo " Translate model: ${TRANSLATE_MODEL}"
echo " Concurrency:     ${MAX_CONCURRENCY}"
echo " Output:          ${OUTPUT_ROOT}"
echo " PDFs:            ${#PDFS[@]}"
echo "========================================"

for entry in "${PDFS[@]}"; do
  IFS='|' read -r pdf_path profile_override <<< "${entry}"

  if [[ ! -f "${pdf_path}" ]]; then
    echo ""
    echo "[SKIP] ${pdf_path} — file not found"
    SKIPPED+=("${pdf_path}")
    continue
  fi

  echo ""
  echo "════════════════════════════════════════"
  echo "  ${pdf_path}"
  if [[ -n "${profile_override}" ]]; then
    echo "  profile: ${profile_override}"
  fi
  echo "════════════════════════════════════════"

  CMD=(
    python -m extraction_pipeline.extract_unified
    --pdf-path "${pdf_path}"
    --output-root "${OUTPUT_ROOT}"
    --published-texts-csv "./input/deep-past-initiative-machine-translation/published_texts.csv"
    --few-shot-path "./data/train_processed.csv"
    --model "${MODEL}"
    --translate-model "${TRANSLATE_MODEL}"
    --max-concurrency "${MAX_CONCURRENCY}"
  )

  if [[ -n "${profile_override}" ]]; then
    CMD+=(--profile "${profile_override}")
  fi

  CMD+=("${EXTRA_ARGS[@]}")

  if "${CMD[@]}"; then
    SUCCEEDED+=("${pdf_path}")
  else
    echo "[FAILED] ${pdf_path}"
    FAILED+=("${pdf_path}")
  fi
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Pipeline complete"
echo "========================================"
echo "  Succeeded: ${#SUCCEEDED[@]}"
echo "  Failed:    ${#FAILED[@]}"
echo "  Skipped:   ${#SKIPPED[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo ""
  echo "  Failed PDFs:"
  for f in "${FAILED[@]}"; do
    echo "    - ${f}"
  done
fi

echo "  Output: ${OUTPUT_ROOT}/"
echo "========================================"
