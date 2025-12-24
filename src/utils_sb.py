# src/utils_sb.py
import ast
import json

def sb_to_name(x):
    """
    StatsBombPy bazen dict döndürür: {"id":..,"name":"Shot"}
    bazen string döndürür: "Shot"
    CSV'de dict-stringleşmiş olabilir: "{'id': 16, 'name': 'Shot'}" veya JSON: {"id":16,"name":"Shot"}
    """
    if x is None:
        return None

    if isinstance(x, dict):
        return x.get("name")

    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None

        # dict gibi görünüyor mu?
        if s.startswith("{") and "name" in s and s.endswith("}"):
            # 1) python dict string
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict) and "name" in d:
                    return d.get("name")
            except Exception:
                pass
            # 2) json string
            try:
                d = json.loads(s)
                if isinstance(d, dict) and "name" in d:
                    return d.get("name")
            except Exception:
                pass

        return s

    return str(x)
