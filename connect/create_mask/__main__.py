import uvicorn

if __name__ == "__main__":
    uvicorn.run("create_mask.main:app", host="0.0.0.0", port=30003, reload=True)
