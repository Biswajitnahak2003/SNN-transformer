# Research Paper Conversion Guide

## Converting Markdown to DOCX with Times New Roman

Your research paper has been created as `research_paper.md`. To convert it to a properly formatted DOCX document with Times New Roman font, follow these steps:

### Method 1: Using Pandoc (Recommended)

1. **Install Pandoc**:
   - Download from: https://pandoc.org/installing.html
   - Or install via Chocolatey: `choco install pandoc`

2. **Convert with proper formatting**:
   ```bash
   pandoc research_paper.md -o research_paper.docx --reference-doc=template.docx
   ```

3. **Create a reference template** (optional but recommended):
   - Open Word and create a new document
   - Set font to Times New Roman 12pt for body text
   - Set headings to Times New Roman 14pt bold
   - Set 1-inch margins, double spacing
   - Save as `template.docx`
   - Use the `--reference-doc=template.docx` option

### Method 2: Using Online Converters

1. **Go to**: https://www.markdowntopdf.com/ or https://cloudconvert.com/md-to-docx
2. **Upload**: `research_paper.md`
3. **Convert**: Download the DOCX file
4. **Format**: Open in Word and apply Times New Roman formatting

### Method 3: Manual Copy-Paste

1. **Open Word** and create a new document
2. **Set formatting**:
   - Font: Times New Roman
   - Size: 12pt for body, 14pt for headings
   - Line spacing: Double
   - Margins: 1 inch all sides

3. **Copy content** from `research_paper.md`
4. **Apply formatting**:
   - Headings: Bold, larger font
   - Abstract: Italic
   - Tables: Convert markdown tables to Word tables
   - Code blocks: Use monospace font

### Method 4: Using VS Code Extensions

1. **Install VS Code extension**: "Markdown PDF" or "Markdown Preview Enhanced"
2. **Open** `research_paper.md` in VS Code
3. **Export** to DOCX or PDF, then convert to DOCX

## Formatting Specifications

- **Font**: Times New Roman throughout
- **Body text**: 12pt
- **Headings**: 14pt bold
- **Abstract**: 12pt italic
- **Captions**: 11pt italic
- **Line spacing**: Double for body text
- **Margins**: 1 inch all sides
- **Page size**: Letter or A4

## Final Output

The converted DOCX should contain:
- Title page with author information
- Abstract with keywords
- 6 main sections (Introduction through Conclusion)
- Tables with results and ablation studies
- References in academic format
- Professional academic formatting

Once converted, your research paper will be ready for submission to conferences or journals!