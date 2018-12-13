#!/usr/bin/env bash
python3 RAISR_train.py
python3 RAISR_test.py
python3 generate_brushstrokes.py
python3 make_dataset.py
