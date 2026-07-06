import re
import json

with open('agents.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update get_combined_context
old_get_combined = '''def get_combined_context(retriever, query: str, max_chunks: int = 20) -> str:
    """Retrieve and format document chunks for a query."""
    if retriever is None:
        return ""
    docs = retriever.invoke(query)
    if not docs:
        return ""
    return "\\n\\n".join([
        f"[{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs[:max_chunks]
    ])'''

new_get_combined = '''def get_combined_context(retriever, query: str, max_chunks: int = 20) -> tuple[str, list]:
    """Retrieve and format document chunks for a query."""
    if retriever is None:
        return "", []
    docs = retriever.invoke(query)
    if not docs:
        return "", []
    combined = "\\n\\n".join([
        f"[{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs[:max_chunks]
    ])
    contexts = [doc.page_content for doc in docs[:max_chunks]]
    return combined, contexts'''

content = content.replace(old_get_combined, new_get_combined)

content = content.replace(
    'combined_content = context if context else get_combined_context(retriever, query)',
    'combined_content, contexts = (context, [context]) if context else get_combined_context(retriever, query)'
)

content = content.replace(
    'return f"No relevant content found in the documents for \'{query}\'."',
    'return json.dumps({"content": f"No relevant content found in the documents for \'{query}\'.", "contexts": []})'
)

# Replace the response returns for the first 5 agents
agents = ['summarizer', 'mcq_generator', 'notes_maker', 'exam_prep_agent', 'concept_explainer']
for agent in agents:
    # Find the end of the agent definition
    # return getattr(response, "content", str(response))
    # It occurs exactly once per agent at the end.
    # We can split the string by `    @tool` and replace in each chunk if it's one of those agents
    pass

parts = content.split('    @tool')
new_parts = [parts[0]]
for part in parts[1:]:
    is_rag = any(f'def {a}(' in part for a in agents)
    if is_rag:
        part = part.replace(
            'return getattr(response, "content", str(response))',
            'return json.dumps({"content": getattr(response, "content", str(response)), "contexts": contexts})'
        )
    new_parts.append(part)

content = '    @tool'.join(new_parts)

with open('agents.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
