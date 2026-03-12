"""
Test coordinate-based table reconstruction from PyMuPDF blocks.
Uses block bounding boxes to group text into rows and columns.
"""
import fitz
from collections import defaultdict

def extract_text_blocks(pdf_path, page_num=0):
    """Extract all text blocks with their bounding box coordinates."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    blocks = page.get_text('dict')['blocks']
    
    text_blocks = []
    for block in blocks:
        if block['type'] == 0:  # text block
            text = ''
            for line in block['lines']:
                for span in line['spans']:
                    text += span['text'] + ' '
            text = text.strip()
            if text and len(text) > 1:
                bbox = block['bbox']  # (x0, y0, x1, y1)
                text_blocks.append({
                    'text': text,
                    'x0': bbox[0],
                    'y0': bbox[1],
                    'x1': bbox[2],
                    'y1': bbox[3],
                    'y_center': (bbox[1] + bbox[3]) / 2,
                    'x_center': (bbox[0] + bbox[2]) / 2,
                })
    
    doc.close()
    return text_blocks


def group_into_rows(blocks, y_tolerance=15):
    """Group text blocks into rows based on Y-coordinate proximity."""
    if not blocks:
        return []
    
    # Sort by Y center
    sorted_blocks = sorted(blocks, key=lambda b: b['y_center'])
    
    rows = []
    current_row = [sorted_blocks[0]]
    
    for block in sorted_blocks[1:]:
        # Check if this block is on the same row (similar Y)
        if abs(block['y_center'] - current_row[-1]['y_center']) <= y_tolerance:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b['x0']))  # Sort by X within row
            current_row = [block]
    
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b['x0']))
    
    return rows


def reconstruct_table(rows, min_cols=3):
    """
    Reconstruct a table from grouped rows.
    Identifies column boundaries from the most common X positions.
    """
    # Filter rows that have enough columns to be table rows
    table_rows = [row for row in rows if len(row) >= min_cols]
    
    if not table_rows:
        return None
    
    # Build table as list of lists
    table_data = []
    for row in table_rows:
        row_texts = [block['text'] for block in row]
        table_data.append(row_texts)
    
    return table_data


def build_card_fee_table(pdf_path):
    """
    Specifically reconstruct the card fee table from the Mashreq KFS PDF.
    Uses a smarter approach: identify the table region, then reconstruct.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # Get all text with positions using 'words' mode for finer granularity
    words = page.get_text('words')  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    
    # Also get blocks for larger text segments
    blocks = page.get_text('dict')['blocks']
    
    # Define the known structure from visual analysis:
    # The table has these columns (approximate X ranges):
    # Col 1: Card Name (x: 38-150)
    # Col 2: Annual Fee Primary (x: 150-300)
    # Col 3: Annual Fee Supplementary (x: 300-420)
    # Col 4: Loyalty (x: 420-800)
    
    # And these row bands (approximate Y ranges):
    # Header: y ~169-190
    # Cashback: y ~200-260
    # noon: y ~265-325
    # Platinum Elite: y ~330-390
    # Solitaire: y ~395-460
    # Platinum Plus: y ~465-540
    
    # Let's use the block data to extract card-specific info
    card_data = {}
    
    # Known card names and their Y-ranges from the block analysis
    card_y_ranges = {
        'Cashback': (200, 265),
        'noon': (265, 330),
        'Platinum Elite': (330, 395),
        'Solitaire': (395, 465),
        'Platinum Plus': (465, 545),
    }
    
    # Column X-ranges
    col_ranges = {
        'Card Name': (0, 150),
        'Annual Fee Primary (VAT inclusive)': (150, 310),
        'Annual Fee Supplementary': (310, 420),
        'Loyalty': (420, 810),
    }
    
    # Extract text for each cell
    for card_name, (y_min, y_max) in card_y_ranges.items():
        card_data[card_name] = {}
        for col_name, (x_min, x_max) in col_ranges.items():
            if col_name == 'Card Name':
                card_data[card_name][col_name] = card_name
                continue
            
            # Find all text blocks in this cell region
            cell_texts = []
            for block in blocks:
                if block['type'] != 0:
                    continue
                by0, by1 = block['bbox'][1], block['bbox'][3]
                bx0, bx1 = block['bbox'][0], block['bbox'][2]
                
                # Check if block overlaps with the cell region
                if by0 < y_max and by1 > y_min and bx0 < x_max and bx1 > x_min:
                    text = ''
                    for line in block['lines']:
                        for span in line['spans']:
                            text += span['text'] + ' '
                    text = text.strip()
                    if text:
                        cell_texts.append(text)
            
            card_data[card_name][col_name] = ' | '.join(cell_texts) if cell_texts else ''
    
    doc.close()
    return card_data


def format_as_markdown(card_data):
    """Convert card data dict to Markdown table."""
    headers = ['Card Name', 'Annual Fee Primary (VAT inclusive)', 'Annual Fee Supplementary', 'Loyalty']
    
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for card_name, data in card_data.items():
        row = [data.get(h, '') for h in headers]
        md += "| " + " | ".join(row) + " |\n"
    
    return md


def format_as_searchable_chunks(card_data, filename):
    """Convert card data to individual searchable chunks per card."""
    chunks = []
    
    # Overview chunk
    overview = "Mashreq Credit Cards - Annual Fees and Loyalty Benefits\n\n"
    for card_name, data in card_data.items():
        overview += f"- {card_name}: Annual Fee = {data.get('Annual Fee Primary (VAT inclusive)', 'N/A')}\n"
    chunks.append({
        'text': overview,
        'chunk_type': 'table_overview',
        'source': filename
    })
    
    # Per-card chunks for precise retrieval
    for card_name, data in card_data.items():
        chunk_text = f"Card: {card_name}\n"
        chunk_text += f"Annual Fee (Primary, VAT inclusive): {data.get('Annual Fee Primary (VAT inclusive)', 'N/A')}\n"
        chunk_text += f"Annual Fee (Supplementary): {data.get('Annual Fee Supplementary', 'N/A')}\n"
        chunk_text += f"Loyalty Benefits: {data.get('Loyalty', 'N/A')}\n"
        chunk_text += f"Source: {filename}"
        
        chunks.append({
            'text': chunk_text,
            'chunk_type': 'card_details',
            'card_name': card_name,
            'source': filename
        })
    
    return chunks


if __name__ == "__main__":
    pdf_path = "/home/ubuntu/upload/Mashreq-Cards-KFS-Final-EngArb.pdf"
    
    print("=" * 80)
    print("TEST 1: Coordinate-based table reconstruction")
    print("=" * 80)
    
    card_data = build_card_fee_table(pdf_path)
    
    print("\nExtracted Card Data:")
    for card, data in card_data.items():
        print(f"\n  {card}:")
        for key, value in data.items():
            print(f"    {key}: {value[:80]}..." if len(value) > 80 else f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Markdown table output")
    print("=" * 80)
    md = format_as_markdown(card_data)
    print(md)
    
    print("\n" + "=" * 80)
    print("TEST 3: Searchable chunks")
    print("=" * 80)
    chunks = format_as_searchable_chunks(card_data, "Mashreq-Cards-KFS.pdf")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({chunk['chunk_type']}) ---")
        print(chunk['text'][:200])
    
    print("\n" + "=" * 80)
    print("TEST 4: Can we answer 'What is the annual fee for PlatinumPlus?'")
    print("=" * 80)
    for chunk in chunks:
        if 'platinum' in chunk['text'].lower() and 'plus' in chunk['text'].lower():
            print(f"\nFOUND in chunk type '{chunk['chunk_type']}':")
            print(chunk['text'])
