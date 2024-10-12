import pandas as pd
from sentence_transformers import SentenceTransformer, util

def main():
    # Cargar el archivo CSV que contiene información sobre las películas de IMDB
    df = pd.read_csv('./IMDB top 1000.csv')

    # Inicializar el modelo que transformará las descripciones de las películas en vectores
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Convertir las descripciones de las películas en vectores (embeddings) en lotes
    embeddings = model.encode(df['Description'], batch_size=64, show_progress_bar=True)
    # Añadir los vectores como una nueva columna en el DataFrame
    df['embeddings'] = embeddings.tolist()

    def compute_similarity(example, query_embedding):
        # Obtener el embedding de la película del DataFrame
        embedding = example['embeddings']
        # Calcular la similitud coseno entre el embedding de la película y el embedding de la consulta
        similarity = util.cos_sim(embedding, query_embedding).item()
        return similarity

    while True:
        # Solicitar el término de búsqueda al usuario
        query = input('Ingresa el término de búsqueda (o escribe "salir" para terminar): ')
        
        # Comprobar si el usuario desea salir
        if query.lower() == "salir":
            print("Saliendo del programa.")
            break
        
        # Convertir la consulta del usuario en un vector (embedding)
        query_embedding = model.encode([query])[0]

        # Aplicar la función de similitud a cada fila del DataFrame y guardar los resultados en una nueva columna 'similarity'
        df['similarity'] = df.apply(lambda x: compute_similarity(x, query_embedding), axis=1)

        # Ordenar el DataFrame por la columna de similitud de mayor a menor
        df = df.sort_values(by='similarity', ascending=False)

        # Mostrar los títulos de las 5 películas más similares a la consulta
        print("Películas más similares a tu búsqueda:")
        print(df.head()['Title'])

        # Agregar una pausa antes de la siguiente iteración
        print("\nPuedes realizar otra búsqueda o escribe 'salir' para terminar.")

# Comprobar si el script se está ejecutando directamente
if __name__ == '__main__':
    main()
