import re
from typing import Optional
import pandas as pd
import requests
from urllib.error import HTTPError
import os

# Source : https://github.com/rexdotsh/spotify-preview-url-workaround
def get_spotify_preview_url(spotify_track_id: str) -> Optional[str]:
    """
    Get the preview URL for a Spotify track using the embed page workaround.

    Args:
        spotify_track_id (str): The Spotify track ID

    Returns:
        Optional[str]: The preview URL if found, else None
    """
    try:
        embed_url = f"https://open.spotify.com/embed/track/{spotify_track_id}"
        response = requests.get(embed_url)
        response.raise_for_status()

        html = response.text
        match = re.search(r'"audioPreview":\s*{\s*"url":\s*"([^"]+)"', html)
        return match.group(1) if match else None

    except Exception as e:
        print(f"Failed to fetch Spotify preview URL: {e}")
        return None
    

def download_preview(preview_url: str, output_filename: str):
    if not preview_url:
        print("No preview available for this track.")
        return

    r = requests.get(preview_url)
    r.raise_for_status()

    with open(output_filename, "wb") as f:
        f.write(r.content)

    print(f"Saved: {output_filename}")
    
def process_csv(csv_path, output_dir="tracks"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "Spotify ID" not in df.columns:
        raise ValueError("CSV must contain a 'Spotify ID' column")

    for track_id in df["Spotify ID"].astype(str):
        track_id = track_id.strip()
        
        if not track_id:
            continue

        print(f"\nProcessing track: {track_id}")

        preview_url = get_spotify_preview_url(track_id)
        
        if not preview_url:
            print("No preview URL found.")
            continue

        filename = os.path.join(output_dir, f"{track_id}.mp3")

        if f"{track_id}.mp3" not in os.listdir(output_dir):

            try : 
                success = download_preview(preview_url, filename)
                print(f"Saved preview: {filename}")
            except HTTPError as e:
                print("HTTP error occurred:", e.code, e.reason)
    
        else :
            print('Track allready downloaded')

if __name__ == "__main__":

       process_csv("emma_spotify.csv")
