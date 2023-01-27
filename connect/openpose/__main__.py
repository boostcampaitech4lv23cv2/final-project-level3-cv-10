import uvicorn

if __name__ == "__main__":
    uvicorn.run("openpose.main:app", host="0.0.0.0", port=30006, reload=True)
