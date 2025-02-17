# SQL-Analytics
# SQL-Analytics
virtual env create: python3 -m venv sql-analytics
activate vm: source sql-analytics/bin/activate
pip3 install -r requirements.txt

start FastAPI:  uvicorn src.app:app --reload

Streamlit on : python -m streamlit run src/streamlit_app.py