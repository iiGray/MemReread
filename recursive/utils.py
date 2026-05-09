import json, re

def subquestions_from_text(response):
    key = 'query'
    try:
        match = re.search(r'<query>(.+)</query>', response, flags = re.DOTALL)
        if match:
            query = match.group(1) 
            query = query.lstrip("{").rstrip("}").strip()
            if len(query) > 5:
                return [query]
    except (ValueError, TypeError):
            pass 
    return []