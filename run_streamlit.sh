#!/bin/bash
export PYTHONPATH="/home/runner/workspace/.pythonlibs/lib/python3.13/site-packages:$PYTHONPATH"
exec streamlit run app.py --server.port 5000
