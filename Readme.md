# Intent_Detection
Intent Detection and Voice Identification for Vietnameses.
Project is built for detection intent in utterances of users and prediction voices.

## To build Docker image
docker build -t voicenlp .

## To run Docker container:
docker run --name voicenlp -p 8000:8000 voicenlp

### URL for add data to model 
1) ./addData (input: {audio:audio_file, username: user_name})
2) ./predict (input: {audio:audio_file})

More docs can be seen in ip_address:8000/docs
