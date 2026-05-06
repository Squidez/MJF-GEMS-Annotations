import re
from typing import Optional
import pandas as pd
import requests
from urllib.error import HTTPError
import os
import yt_dlp
from yt_dlp.utils import download_range_func
 

def download_youtube_excerpt (clip_url: str, output_dir: str):
    """
    Download the audio of a youtube video
    
    Args:
        clip_url (str): A Youtube URL
        output_dir (str): The output Folder
    """

    yt_url = clip_url
    id = clip_url[24:] # extract the id from the url

    # Define parameters
    yt_opts = {
    'verbose': True,
    'format': 'm4a/bestaudio/best',
    'outtmpl': f'{output_dir}/%(title).%(ext)s',
    "download_sections": ["*0-300"], 
    'force_keyframes_at_cuts': True,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        }]
    }

    # Dowlnoad the audio of the url
    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        ydl.download(yt_url)

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
    """
    Download the Spotify Preview

    Args:
        preview_url (str) : The Spotify Preview URL
        output_filename (str): The output file name
    """

    if not preview_url:
        print("No preview available for this track.")
        return

    # Get the URL content
    r = requests.get(preview_url)
    r.raise_for_status()

    # Write the URL content into a file
    with open(output_filename, "wb") as f:
        f.write(r.content)

    print(f"Saved: {output_filename}")


def process_csv(csv_path, output_dir="tracks"):
    """
    Download the tracks from a csv file
    
    Args:
        csv_path (str) : Path  of the csv file
        output_dir (str): Output Directory
    """

    # Create tracks folder
    os.makedirs(output_dir, exist_ok=True)

    # Open csv file
    df = pd.read_csv(csv_path,
                 delimiter=',',
                 encoding='cp1252',
                 na_values='NA')
    # create new path column if it dosen't exist
    if 'Path' not in df.columns:
        df['Path'] = pd.NA

    # Iter the rows of the csv
    for index, row in df.iterrows():
        
        track_id = row['Youtube ID'] 

        if f"{track_id}.mp3" not in os.listdir(output_dir):

            # Try to download the youtube video
            try :
                download_youtube_excerpt(row['Youtube Clip'], output_dir)

                # Add the path to the csv
                df.at[index,'Path'] = f'{output_dir}/{track_id}.mp3'
                df.to_csv('emma_final.csv', encoding='cp1252')

            except:

                print("\nCouldn't Download Youtube Clip -- Trying With Spotify ...")
                
                # Try to download the Spotify Preview
                track_id = row['Spotify ID']
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

                        # Add the path to the csv
                        df.at[index,'Path'] = f'{output_dir}/{track_id}.mp3'
                        df.to_csv('emma_final.csv', encoding='cp1252')

                    except HTTPError as e:
                        print("HTTP error occurred:", e.code, e.reason)
        
                else :
                    print('Track allready downloaded')
            
        else:
            print('Track allready downloaded')


if __name__ == "__main__":

    process_csv("EMMA__filtered.csv")