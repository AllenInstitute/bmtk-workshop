#!/bin/bash
set -e

jupyter lab --allow-root --ip=* --port 8888 --no-browser --notebook-dir /home/shared/bmtk-workshop --NotebookApp.token=""
