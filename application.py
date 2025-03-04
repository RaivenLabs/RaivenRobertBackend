from app import create_app

application = create_app()

app = application

@app.route('/')  # Health check endpoint for Elastic Beanstalk Monitoring
def health():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    application.run()
