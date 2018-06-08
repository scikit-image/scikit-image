#!/bin/bash
# Accepts a .pyx file path and deletes an associated .c (and md5) file,
# if present.
filename="${1%.*}".c
if [ -e "$filename" ]; then
	rm -f "$filename";
fi
filename="${1%.*}".md5
if [ -e "$filename" ]; then
	rm -f "$filename";
fi
