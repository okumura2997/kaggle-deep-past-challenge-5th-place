#!/bin/sh

python src/extract_published_texts_akt_subset.py
python src/find_excavation_number_pages.py
python src/extract_excavation_transliteration_pairs.py
python src/extract_excavation_translation_pairs.py
python src/preprocess_extracted_data.py
python src/translate.py