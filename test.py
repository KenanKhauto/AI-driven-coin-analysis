import re
filename = "00012a00" 

patterns = [
    r"(.*?)_a$",        # Matches filenames ending in "_a"
    r"(.*?)_r$",        # Matches filenames ending in "_r"
    r"(.*?)_av$",       # Matches filenames ending in "_av"
    r"(.*?)_rv$",       # Matches filenames ending in "_rv"
    r"(.*?)_obv$",      # Matches filenames ending in "_obv"
    r"(.*?)_rev$",      # Matches filenames ending in "_rev"
    r"(.*?)a\d+$",      # Matches filenames like "00012a00"
    r"(.*?)r\d+$",      # Matches filenames like "00012r00"
]

for pattern in patterns:
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        print(f"Testing pattern: {pattern}")
        print(f"Match: {match.group(0)}")
        cleaned_filename = re.sub(pattern, r"\1", filename, flags=re.IGNORECASE)
        print(f"Cleaned filename: {cleaned_filename}")