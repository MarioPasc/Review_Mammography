#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_longtable.py

This script reads a LaTeX file containing a 'longtable' environment and
extracts its contents into a CSV file with identical columns.

It:
  1. Locates the \begin{longtable}…\end{longtable} block.
  2. Splits into rows on '\\\\' (row terminator).
  3. Splits each row on '&' into fields.
  4. Strips out common LaTeX commands and braces.
  5. Writes a CSV with the same header.
"""

import re
import csv
import argparse

def clean_latex(text):
    """
    Remove simple LaTeX commands and braces:
      - \\command{X}        → X
      - remaining { }       → removed
      - multiple spaces     → single space
    """
    # 1. Replace \command{…} or \command*{…} with the contents
    text = re.sub(r'\\[a-zA-Z]+\*?{([^}]*)}', r'\1', text)
    # 2. Remove any remaining braces
    text = text.replace('{', '').replace('}', '')
    # 3. Collapse whitespace
    return ' '.join(text.split())

def extract_rows(tex):
    """
    Given the full .tex text, return a list of rows (each a list of fields).
    Skips:
      - lines with only \hline or LaTeX-head directives
      - repeated header blocks (\textbf…)
      - multicolumn/footer directives
    """
    # 1. Isolate the longtable
    m = re.search(r'\\begin\{longtable\}.*?\\hline(.*?)\\end\{longtable\}', 
                  tex, re.DOTALL)
    if not m:
        raise ValueError("No longtable environment found.")
    body = m.group(1)

    # 2. Split into raw rows on "\\" (LaTeX row terminator)
    raw_rows = re.split(r'\\\\', body)
    rows = []
    for raw in raw_rows:
        line = raw.strip()
        # Skip empty lines or pure-Latex directives
        if not line or line.startswith('%') or line.startswith('\\') or 'multicolumn' in line:
            continue
        # Skip repeated header
        if line.lower().startswith(r'\textbf'):
            # Header we'll capture once, below
            rows.append([f.strip() for f in line.split('&')])
            continue
        # 3. Split into columns
        fields = [clean_latex(f) for f in line.split('&')]
        rows.append(fields)
    return rows

def write_csv(rows, out_path):
    """
    Write rows (list of lists) to CSV.
    Assumes rows[0] is the header.
    """
    with open(out_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        for row in rows:
            writer.writerow(row)

def main():
    
    texfile = "data/tex/longtable.tex"
    csvfile = "data/csvs/longtable.csv"

    with open(texfile, 'r', encoding='utf-8') as f:
        tex = f.read()

    rows = extract_rows(tex)
    write_csv(rows, csvfile)

if __name__ == '__main__':
    main()
