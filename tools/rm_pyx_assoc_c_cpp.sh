#!/bin/bash
# Accepts a .pyx file path and deletes an associated .c file, if present.
filename="${1%.*}".c
if [ -e "$filename" ]; then
	rm -f "$filename";
fi

filename="${1%.*}".cpp
if [ -e "$filename" ]; then
	rm -f "$filename";
fi
