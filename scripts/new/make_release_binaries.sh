#!/bin/bash
set -e

# https://github.com/neurobin/shc
shc -U -f ids.sh -o ../../ids
shc -U -f capture.sh -o ../../capture
shc -U -f detect_intrusions.sh -o ../../detect_intrusions

rm ids.sh.x.c capture.sh.x.c detect_intrusions.sh.x.c 