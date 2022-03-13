FROM pablohdez98/dbpedia-abstracts-to-rdf:latest

WORKDIR /app

COPY . .

CMD ["python", "code/app/app.py", "--input_text=code/app/input.txt",  "--all_sentences", "--save_debug" ]