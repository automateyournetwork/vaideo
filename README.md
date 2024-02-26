# vaideo
Create AI Generated Voice Overs for your videos

![Vaideo Logo](/vaideo/vaideo.png)

## Getting started 

Git Clone the repository 

Create a .env file inside the /vaideo/vaideo folder (where the main files are located) with your OPENAI_API_KEY

```console
OPENAI_API_KEY="<your key here>"
```

This uses chatGPT 4 Vision and Text to Speech and you must have an API key with access to these models. 

## Docker-compose 
This was build using Windows Sub-systems for Linux (WSL2), Ubunutu, and Docker Desktop 

It should work on other platforms that support docker-compose 

```console
~/vaideo$ docker-compose up
```

## Streamlit
Visit http://localhost:8585 to launch Vaideo 

## Voice Over
Proceed to the Voice Over Page

Upload your 5 - 30 second video or select a video from YouTube

Select a voice 

*optional - choose a conversation starter

Add your personal touch to the Voice Over Script 

Generate the voice over 

## Cartoon
Proceed to the Cartoon Page

Upload your 5 - 30 second video or select a video from YouTube

Select a voice 

*optional - choose a conversation starter

Add your personal touch to the Image Generation Script 

Generate the cartoon

