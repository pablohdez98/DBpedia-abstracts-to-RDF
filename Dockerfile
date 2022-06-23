FROM pablohdez98/dbpedia-abstracts-to-rdf:latest

WORKDIR /app

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "code/webapp/app.py"]