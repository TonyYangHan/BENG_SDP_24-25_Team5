from XY_cycle import XY_cycle
import os, datetime as dt, pytz as tz, time, argparse, gc
from spin_motor import spin_start, spin_stop
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

parser = argparse.ArgumentParser(description = "Please enter a command, -h for help")

parser.add_argument("duration", type = int, help = "The total number of hours of observation")
parser.add_argument("interval", type = int, help = "The interval between each photo upload in seconds")
parser.add_argument("-o", help = "Output directory of organoid data")

args = parser.parse_args()

duration = args.duration * 3600
interval = args.interval


SCOPES = ['https://www.googleapis.com/auth/drive']

# folder = "1kGmciW9RG8USL8sBMpX3glfaHrE_F_Ov" # Main folder for 24-25 images
folder = "1QsqQqV88MaevSBf_SdGpIzr2WKunXwcZ" # Folde for 24 hr streaming experiments

def get_refreshed_credentials(SCOPES):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    
    elif not creds:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server()

    with open('token.json', 'w') as token_file:
        token_file.write(creds.to_json())
            
    return creds

start_time = time.time()

# Upload photos at a given interval for some duration
while  time.time() - start_time < duration:

    spin_stop()

    time.sleep(20) # Let organoids settle to the bottom of the plate after spin motor stops

    XY_cycle()

    for ip in sorted(os.listdir("temp_img_cache")):
        full_path = "temp_img_cache/" + ip
        creds = get_refreshed_credentials(SCOPES)

        service = build('drive', 'v3', credentials=creds)

        curr_time = dt.datetime.now(tz.timezone("US/Pacific"))

        file_meta = {'name': f"{ip}_{str(curr_time)}.jpg", "parents": [folder]}

        media = MediaFileUpload(full_path)
        file = service.files().create(body=file_meta,
                                            media_body=media,
                                            fields='id').execute()
    
        os.remove(full_path)
    
    spin_start() # restart spin motors

    gc.collect() # collect garbage in memory to ensure the program runs for extended periods of time

    time.sleep(interval)

print("Streaming stopped.")
