from XY_Cycle import XYCycle
from picamera2 import Picamera2
import os, datetime as dt, pytz as tz, time, argparse, json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

parser = argparse.ArgumentParser(description = "Please enter a command")

parser.add_argument("duration", type = int, required = True, help = "The total number of hours of observation")
parser.add_argument("interval", type = int, help = "The interval between each photo upload")
parser.add_argument("-o", help = "Output directory of organoid data")

args = parser.parse_args()

duration = args.duration * 3600
if args.interval == None:
    interval = 3600


SCOPES = ['https://www.googleapis.com/auth/drive']

folder = "1kGmciW9RG8USL8sBMpX3glfaHrE_F_Ov"

def get_refreshed_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open('token.json', 'w') as token_file:
            token_file.write(creds.to_json())
    
    elif not creds:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_console()
        # Save credentials
        with open('token.json', 'w') as token:
            json.dump({
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }, token)
    return creds

start_time = time.time()

# Upload photos at a given interval for some duration
# Need to change code to refresh token every hour
while  time.time() - start_time < duration:

    XYCycle()

    for ip in sorted(os.listdir("../temp_img_cache")): # Replace with actual directory
        creds = get_refreshed_credentials()

        service = build('drive', 'v3', credentials=creds)

        curr_time = dt.datetime.now(tz.timezone("US/Pacific"))

        file_meta = {'name': f"{str(curr_time)}.jpg", "parents": [folder]}

        media = MediaFileUpload(ip)
        file = service.files().create(body=file_meta,
                                            media_body=media,
                                            fields='id').execute()
    
        os.remove(ip)
    
    time.sleep(interval)

print("Streaming stopped.")
