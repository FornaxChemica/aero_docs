from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from embedchain import App
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


load_dotenv()
app = FastAPI()
embedchain_app = App.from_config(config_path="config.yaml")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

class Item(BaseModel):
    file: str
    videoUrls: str | None = None
    title: str | None = None
    comments: str | None = None

@app.post("/upload_pdf")
async def create_item(item: Item):
    video_urls = json.loads(item.videoUrls)
    first_video = next((item for item in video_urls if item["type"].lower().startswith("video")), None)
    if first_video is None:
        first_video = {"url": ""}

    
    embedchain_app.add(item.file, data_type='pdf_file', metadata={"video_url": first_video['url'], "title": item.title, "comments": item.comments})
    embedchain_app.add(item.title + " " + item.comments, data_type='text', metadata={"video_url": first_video['url'], "title": item.title, "comments": item.comments})

    all_images = [item["url"] for item in video_urls if item["type"].lower().startswith("image")]

    print(all_images)

    responses = await asyncio.gather(*(upload_image(image) for image in all_images))

    for i, response in enumerate(responses):
        embedchain_app.add(response, data_type='text', metadata={"source": all_images[i], "type": "image", "title": item.title, "comments": item.comments, "video_url": first_video['url']})

    return {"file_name": item.file}

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    return FileResponse("files/" + file_name)

@app.get("/ask_question")
async def ask_question(message: str, type: str = "", imageUrls: str = ""):
    """Ask a question to the AI model."""
    imageDescriptions = json.loads(imageUrls)
    responses = await asyncio.gather(*(upload_image(image) for image in imageDescriptions))
    if len(type) > 0:
        answer, sources = embedchain_app.chat(message + "\n\n" +json.dumps(responses), citations=True, where={"type": type})
    else:
        answer, sources = embedchain_app.chat(message + "\n\n" + json.dumps(responses), citations=True)

    return {"answer": answer, "sources": sources}

async def upload_image(image: str) -> str:
    """Uploads an image and gets a response."""

    # Send the request and get the response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    },
                ],
            }
        ],
    )
    print(response.choices)
    return response.choices[0].message.content