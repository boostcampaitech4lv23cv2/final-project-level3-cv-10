import uvicorn

if __name__ == "__main__":
    uvicorn.run("create_hp.main:app", host="0.0.0.0", port=30005, reload=True)
