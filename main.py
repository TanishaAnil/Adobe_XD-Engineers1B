import os
import json
import datetime
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

input_dir = "input"
output_dir = "output"

# Load persona and job
with open(os.path.join(input_dir, "persona_job.json"), "r", encoding="utf-8") as f:
    persona_job = json.load(f)

persona_text = persona_job["persona"] + " " + persona_job["job_to_be_done"]

# Load local model (must be downloaded beforehand)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get embeddings of persona-task
persona_embedding = model.encode(persona_text, convert_to_tensor=True)

extracted_sections = []
subsection_analysis = []

# Recursively collect PDF file paths
pdf_filepaths = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            pdf_filepaths.append(full_path)

# Prepare relative filenames for metadata
input_docs = [os.path.relpath(path, input_dir) for path in pdf_filepaths]

# Process each PDF
for filepath in pdf_filepaths:
    filename = os.path.relpath(filepath, input_dir)
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        print(f"⚠️ Failed to open {filename}: {e}")
        continue

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        if len(text.strip()) < 100:
            continue  # skip empty or low-text pages

        # break into paragraph-level chunks
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

        if not paragraphs:
            continue

        # embed paragraphs and compare
        para_embeddings = model.encode(paragraphs, convert_to_tensor=True)
        similarities = util.cos_sim(persona_embedding, para_embeddings)[0]

        # Get top 2 paragraphs
        top_indices = similarities.argsort(descending=True)[:2]
        for idx in top_indices:
            para_text = paragraphs[idx]
            sim_score = similarities[idx].item()

            extracted_sections.append({
                "document": filename,
                "section_title": para_text[:80] + ("..." if len(para_text) > 80 else ""),
                "importance_rank": 0,  # temp
                "page_number": page_num + 1
            })

            subsection_analysis.append({
                "document": filename,
                "refined_text": para_text,
                "page_number": page_num + 1
            })

# Rank top 5 extracted sections globally
extracted_sections.sort(
    key=lambda x: next(
        p['refined_text'] for p in subsection_analysis
        if p['document'] == x['document'] and p['page_number'] == x['page_number']
    ).__len__(),
    reverse=True
)

for i, section in enumerate(extracted_sections[:5]):
    section["importance_rank"] = i + 1

# Keep only top 5
extracted_sections = extracted_sections[:5]
top_docs_and_pages = {(s["document"], s["page_number"]) for s in extracted_sections}
subsection_analysis = [
    s for s in subsection_analysis
    if (s["document"], s["page_number"]) in top_docs_and_pages
]

# Build metadata
output = {
    "metadata": {
        "input_documents": input_docs,
        "persona": persona_job["persona"],
        "job_to_be_done": persona_job["job_to_be_done"],
        "processing_timestamp": datetime.datetime.now().isoformat()
    },
    "extracted_sections": extracted_sections,
    "subsection_analysis": subsection_analysis
}

# Save final output
output_path = os.path.join(output_dir, "output.json")
os.makedirs(output_dir, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(output, out_file, indent=4, ensure_ascii=False)

print(f"✅ Output saved to {output_path}")
