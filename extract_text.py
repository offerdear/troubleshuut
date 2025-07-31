import fitz  # PyMuPDF

# Open PDF
pdf_path = "/Users/henrylee/Projects-DaveClassJuly/returnOrderApp/troubleshuut-main/MacBookProUserGuide.pdf"
doc = fitz.open(pdf_path)

# Extract text from each page
text = ""
for page in doc:
    text += page.get_text() + "\n"

# Save text to file
with open("MacBookPro_user_manual.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Extraction complete!")
