def chunk(reasoning: str, granularity: int = 20):
    """
    Chunk the reasoning into smaller chunks.
    """
    chunks = reasoning.split('\n\n')
    masks = [len(chunk.split()) > granularity for chunk in chunks]
    
    # Step 1: chunk the sequence into small chunks
    merged, buffer = [], []
    for c, m in zip(chunks, masks):
        if not m:
            buffer.append(c)
        else:
            if buffer:
                merged.append('\n\n'.join(buffer))
                buffer.clear()
            merged.append(c)
    if buffer:
        merged.append('\n\n'.join(buffer))
    
    # Step 2: merge small chunks to big chunks
    super_chunks, current = [], None
    for c in merged:
        if len(c.split()) > granularity:
            if current is not None:
                super_chunks.append(current)
            current = c
        else:
            if current is None:
                current = c
            else:
                current += '\n\n' + c
    
    if current is not None:
        super_chunks.append(current)
    
    return super_chunks