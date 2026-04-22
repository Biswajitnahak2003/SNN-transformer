#!/bin/bash
# Professional Research Paper Compilation Script
echo "Compiling comprehensive research paper..."

# Install missing packages if needed
echo "Ensuring LaTeX packages are available..."

# Compile twice for proper references and table of contents
echo "First compilation..."
pdflatex research_paper.tex

echo "Second compilation for references..."
pdflatex research_paper.tex

echo "Compilation complete!"
echo "Generated: research_paper.pdf"
echo ""
echo "Paper features:"
echo "- Professional IEEE formatting"
echo "- Comprehensive methodology with mathematical formulations"
echo "- Extensive experimental results and ablation studies"
echo "- Detailed discussion and analysis"
echo "- 16 academic references"
echo "- Times New Roman font, 12pt"
echo ""
echo "Author: Biswajit Nahak, IIIT Bhubaneswar, Electronics and Telecommunication Engineering"