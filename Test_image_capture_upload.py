# from picamera2 import Picamera2


# # Initialize Picamera2
# picam2 = Picamera2()

# # Configure the camera for still image capture
# camera_config = picam2.create_still_configuration()
# picam2.configure(camera_config)

# # Start the camera
# picam2.start()

# # Capture an image
# picam2.capture_file("test_image.jpg")

# # Stop the camera
# picam2.stop()

# print("Image captured and saved as image.jpg!")


import os, json, time
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/drive']

creds = None

if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)

# if not creds or not creds.valid:
#     if creds and creds.expired and creds.refresh_token:
#         creds.refresh()
#     else:
#         flow = InstalledAppFlow.from_client_secrets_file(
#             'credentials.json', SCOPES)
#         creds = flow.run_console()
#     # Save credentials
#     with open('token.json', 'w') as token:
#         json.dump({
#             'token': creds.token,
#             'refresh_token': creds.refresh_token,
#             'token_uri': creds.token_uri,
#             'client_id': creds.client_id,
#             'client_secret': creds.client_secret,
#             'scopes': creds.scopes
#         }, token)

service = build('drive', 'v3', credentials=creds)

img_dir = './test_image.jpg'
folder = "1kGmciW9RG8USL8sBMpX3glfaHrE_F_Ov"

start_time = time.time()
for i in range(10):
    file_meta = {'name': "test_image_" + str(i) + ".jpg",
                "parents": [folder]
                }

    media = MediaFileUpload(img_dir)
    file = service.files().create(body=file_meta,
                                        media_body=media,
                                        fields='id').execute()

end_time = time.time()

print(end_time - start_time)