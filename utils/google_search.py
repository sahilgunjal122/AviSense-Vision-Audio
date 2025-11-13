import os
import requests

def get_bird_info(query, api_key, cse_id):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
    response = requests.get(url)
    data = response.json()

    if "items" in data:
        for item in data["items"]:
            title = item.get("title")
            snippet = item.get("snippet")
            link = item.get("link")
            pagemap = item.get("pagemap")
            if pagemap and "cse_image" in pagemap:
                image = pagemap["cse_image"][0].get("src")
            else:
                image = None
            return {"title": title, "description": snippet, "link": link, "image": image}

    return None
