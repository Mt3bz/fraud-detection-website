#!/bin/bash
pip install --upgrade pip setuptools wheel
pip install numpy==1.21.6 --only-binary=:all:
pip install -r requirements.txt
