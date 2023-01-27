import uvicorn

if __name__ == "__main__":
    uvicorn.run("densepose.main:app", host="0.0.0.0", port=30004, reload=True)
