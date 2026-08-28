## AI Research Paper Assistant

### Run the backend

```powershell
uvicorn backend.main:app --reload
```

### Run the frontend

Install `requirements.txt`, then start Streamlit from the project root:

```powershell
streamlit run frontend/app.py
```

Set `PAPER_API_URL` when the API is not running at `http://localhost:8000`.
