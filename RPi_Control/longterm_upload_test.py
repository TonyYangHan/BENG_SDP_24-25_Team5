from picamera2 import Picamera2
import os, datetime as dt, pytz as tz, time, json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def get_refreshed_credentials(SCOPES):
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


# Initialize Picamera2
picam2 = Picamera2()

# Configure the camera for still image capture
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)

# Start the camera
picam2.start()


# Define the scope
SCOPES = ['https://www.googleapis.com/auth/drive']
img_dir = './test_image.jpg'
folder = "1kGmciW9RG8USL8sBMpX3glfaHrE_F_Ov"
duration = 48*3600
interval = 3600

start_time = time.time()

while  time.time() - start_time < duration:

    creds = get_refreshed_credentials(SCOPES)

    service = build('drive', 'v3', credentials=creds)

    picam2.capture_file("test_image.jpg")

    curr_time = dt.datetime.now(tz.timezone("US/Pacific"))

    file_meta = {'name': f"{str(curr_time)}.jpg",
                "parents": [folder]
                }

    media = MediaFileUpload(img_dir)
    file = service.files().create(body=file_meta,
                                        media_body=media,
                                        fields='id').execute()
    
    os.remove("test_image.jpg")
    
    time.sleep(interval)

picam2.stop()