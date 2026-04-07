from customer_support_env.server import app

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
